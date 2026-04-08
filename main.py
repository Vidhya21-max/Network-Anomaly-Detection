"""
Entry point for the Real-Time Network Anomaly Detection System.
Trains (or loads) model, computes threshold, runs streaming simulation with optional logging.
"""
import argparse
import logging
import sys
from pathlib import Path

from config import (
    LOGS_DIR,
    LOG_FILE,
    LOG_LEVEL,
    MODEL_DIR,
    DATASET_PATH,
    STREAM_BATCH_SIZE,
    STREAM_DELAY_SECONDS,
    ISOLATION_FOREST_PATH,
    NORMAL_LABEL,
)
from db.store import init_db, save_anomaly
from alerts.notify import send_alert, AlertConfig
from data.preprocess import load_data, prepare_data
from data.generate_sample_data import generate_sample_network_data
from models.autoencoder import (
    train_autoencoder,
    reconstruction_errors,
    save_model,
    load_model,
)
from models.isolation_forest import (
    train_isolation_forest,
    anomaly_scores as iforest_anomaly_scores,
    save_iforest,
    load_iforest,
)
from detector import AnomalyDetector, IsolationForestDetector
from streaming.simulator import StreamSimulator


def setup_logging():
    """Configure logging to file and console."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("anomaly_detection")


def main():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection System")
    parser.add_argument("--train", action="store_true", help="Train Autoencoder (and Isolation Forest) and save")
    parser.add_argument("--train-if", action="store_true", help="Train only Isolation Forest and save")
    parser.add_argument("--tune", action="store_true", help="Use Optuna to tune Autoencoder hyperparameters")
    parser.add_argument("--stream", type=int, default=0, metavar="N", help="Run N batches of stream (0 = no stream)")
    parser.add_argument("--both", action="store_true", help="Use both Autoencoder and Isolation Forest when streaming")
    parser.add_argument("--no-delay", action="store_true", help="Disable delay between stream batches")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample dataset if missing")
    parser.add_argument("--evaluate", action="store_true", help="Print Precision/Recall/F1/ROC when labels available")
    parser.add_argument("--save-db", action="store_true", help="Save detected anomalies to SQLite during stream")
    parser.add_argument("--alert", action="store_true", help="Send alerts (email if configured) for high-severity anomalies")
    args = parser.parse_args()

    logger = setup_logging()
    init_db()

    # Ensure data exists
    if not DATASET_PATH.exists():
        if args.generate_data:
            generate_sample_network_data()
            logger.info("Generated sample dataset at %s", DATASET_PATH)
        else:
            logger.warning("Dataset not found. Run with --generate-data or place data at %s", DATASET_PATH)
            generate_sample_network_data()
            logger.info("Generated sample dataset for demo.")

    df = load_data()
    prep = prepare_data(df)
    X_train = prep["X_train_norm"]
    X_test = prep["X_test"]
    feature_names = prep["feature_names"]
    logger.info("Data loaded: train %d, test %d, features %d", len(X_train), len(X_test), len(feature_names))

    # ---- Autoencoder ----
    model_path = MODEL_DIR / "autoencoder"
    if args.tune:
        from tuning.optuna_tune import run_optuna_tuning
        logger.info("Starting Optuna tuning...")
        autoencoder, best_params, _ = run_optuna_tuning(X_train, n_trials=10, verbose=1)
        logger.info("Best params: %s", best_params)
        save_model(autoencoder)
        model = autoencoder
    elif model_path.exists() and not args.train:
        try:
            model = load_model()
            logger.info("Loaded saved Autoencoder from %s", model_path)
        except Exception as e:
            logger.warning("Could not load Autoencoder: %s. Training new model.", e)
            autoencoder, _, _ = train_autoencoder(X_train, verbose=1)
            save_model(autoencoder)
            model = autoencoder
    else:
        autoencoder, _, _ = train_autoencoder(X_train, verbose=1)
        save_model(autoencoder)
        model = autoencoder
        logger.info("Autoencoder trained and saved to %s", model_path)

    train_errors = reconstruction_errors(model, X_train)
    detector = AnomalyDetector(model)
    detector.set_threshold_from_train_errors(train_errors)
    logger.info("Autoencoder threshold: %.6f", detector.threshold)

    # ---- Isolation Forest ----
    detector_if = None
    if args.train or args.train_if:
        if_model = train_isolation_forest(X_train)
        save_iforest(if_model)
        logger.info("Isolation Forest trained and saved to %s", ISOLATION_FOREST_PATH)
        detector_if = IsolationForestDetector(if_model)
        detector_if.set_threshold_from_train_errors(iforest_anomaly_scores(if_model, X_train))
        logger.info("Isolation Forest threshold: %.6f", detector_if.threshold)
    elif ISOLATION_FOREST_PATH.exists():
        try:
            if_model = load_iforest()
            detector_if = IsolationForestDetector(if_model)
            detector_if.set_threshold_from_train_errors(iforest_anomaly_scores(if_model, X_train))
            logger.info("Loaded Isolation Forest from %s", ISOLATION_FOREST_PATH)
        except Exception as e:
            logger.warning("Could not load Isolation Forest: %s", e)

    # Streaming
    if args.stream > 0:
        delay = 0 if args.no_delay else STREAM_DELAY_SECONDS
        sim = StreamSimulator(
            X_test,
            detector,
            batch_size=STREAM_BATCH_SIZE,
            delay_seconds=delay,
            detector_2=detector_if if (args.both and detector_if is not None) else None,
        )
        alert_cfg = AlertConfig() if args.alert else None
        n = 0
        from datetime import datetime
        for result in sim:
            n += 1
            for i, (is_anom, score, sev) in enumerate(zip(result["is_anomaly"], result["scores"], result["severities"])):
                if is_anom:
                    logger.info("AE anomaly #%d: score=%.4f severity=%s", detector.anomaly_count, score, sev)
                    if args.save_db:
                        save_anomaly(datetime.utcnow().isoformat() + "Z", "autoencoder", float(score), sev)
                    if args.alert and sev == "High":
                        send_alert(sev, float(score), "Stream detection", alert_cfg)
            if "is_anomaly_2" in result:
                for i, (is_anom, score, sev) in enumerate(zip(result["is_anomaly_2"], result["scores_2"], result["severities_2"])):
                    if is_anom:
                        logger.info("IF anomaly: score=%.4f severity=%s", score, sev)
                        if args.save_db:
                            save_anomaly(datetime.utcnow().isoformat() + "Z", "isolation_forest", float(score), sev)
                        if args.alert and sev == "High":
                            send_alert(sev, float(score), "Stream detection (IF)", alert_cfg)
            if n >= args.stream:
                break
        logger.info("Stream finished. AE anomalies: %d; IF anomalies: %s",
                   detector.anomaly_count, detector_if.anomaly_count if detector_if else "N/A")

    if args.evaluate and prep.get("has_labels") and prep.get("y_test") is not None:
        from utils.metrics import compute_metrics, compute_roc
        y_true = (prep["y_test"] != NORMAL_LABEL).astype(int)
        scores_ae = detector.predict_scores(X_test)
        th_ae = detector.threshold or 0.0
        y_pred_ae = (scores_ae > th_ae).astype(int)
        m = compute_metrics(y_true, y_pred_ae)
        roc = compute_roc(y_true, scores_ae)
        logger.info("Evaluation (Autoencoder): Precision=%.4f Recall=%.4f F1=%.4f ROC AUC=%.4f",
                    m["precision"], m["recall"], m["f1"], roc["auc"])
        if detector_if is not None:
            scores_if = detector_if.predict_scores(X_test)
            th_if = detector_if.threshold or 0.0
            y_pred_if = (scores_if > th_if).astype(int)
            m_if = compute_metrics(y_true, y_pred_if)
            roc_if = compute_roc(y_true, scores_if)
            logger.info("Evaluation (Isolation Forest): Precision=%.4f Recall=%.4f F1=%.4f ROC AUC=%.4f",
                        m_if["precision"], m_if["recall"], m_if["f1"], roc_if["auc"])

    logger.info("Done. Start dashboard: streamlit run dashboard/app.py | API: uvicorn api.app:app")


if __name__ == "__main__":
    main()
