"""
Streamlit dashboard: Autoencoder vs Isolation Forest comparison.
Live anomaly score graph, score over time, anomaly count, packet table, XAI (Autoencoder only).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DASHBOARD_REFRESH_INTERVAL,
    DASHBOARD_MAX_PACKETS_TABLE,
    MODEL_DIR,
    DATASET_PATH,
    ISOLATION_FOREST_PATH,
)
from data.preprocess import load_data, prepare_data
from data.generate_sample_data import generate_sample_network_data
from models.autoencoder import train_autoencoder, reconstruction_errors, load_model, save_model
from models.isolation_forest import (
    train_isolation_forest,
    anomaly_scores as iforest_anomaly_scores,
    load_iforest,
    save_iforest,
)
from detector import AnomalyDetector, IsolationForestDetector
from streaming.simulator import StreamSimulator
from xai.explain import top_contributing_features
from utils.metrics import compute_metrics, compute_roc
from config import NORMAL_LABEL


def ensure_data():
    """Load or generate dataset."""
    if DATASET_PATH.exists():
        return load_data()
    return generate_sample_network_data()


def _threshold_from_scores(scores, k=3.0):
    return float(np.mean(scores) + k * np.std(scores)) if len(scores) else 0.0


@st.cache_resource
def get_models_and_data():
    """
    Load or train Autoencoder and Isolation Forest; return both detectors (IF may be None).
    Returns: detector_ae, detector_if (or None), feature_names, X_test, scaler, y_test.
    """
    data_df = ensure_data()
    prep = prepare_data(data_df)
    X_train = prep["X_train_norm"]
    X_test = prep["X_test"]
    feature_names = prep["feature_names"]
    scaler = prep["scaler"]
    y_test = prep.get("y_test")

    # ---- Autoencoder ----
    model_path = MODEL_DIR / "autoencoder"
    if model_path.exists():
        try:
            ae_model = load_model()
        except Exception:
            ae_model = None
        else:
            train_errors = reconstruction_errors(ae_model, X_train)
            th = _threshold_from_scores(train_errors)
            detector_ae = AnomalyDetector(ae_model, threshold=th)
            detector_ae.anomaly_count = 0
    else:
        ae_model = None
    if ae_model is None:
        autoencoder, _, _ = train_autoencoder(X_train, epochs=30, verbose=0)
        save_model(autoencoder)
        train_errors = reconstruction_errors(autoencoder, X_train)
        th = _threshold_from_scores(train_errors)
        detector_ae = AnomalyDetector(autoencoder, threshold=th)

    # ---- Isolation Forest ----
    detector_if = None
    if ISOLATION_FOREST_PATH.exists():
        try:
            if_model = load_iforest()
            train_scores = iforest_anomaly_scores(if_model, X_train)
            th = _threshold_from_scores(train_scores)
            detector_if = IsolationForestDetector(if_model, threshold=th)
            detector_if.anomaly_count = 0
        except Exception:
            pass
    if detector_if is None:
        if_model = train_isolation_forest(X_train)
        save_iforest(if_model)
        train_scores = iforest_anomaly_scores(if_model, X_train)
        th = _threshold_from_scores(train_scores)
        detector_if = IsolationForestDetector(if_model, threshold=th)

    return detector_ae, detector_if, feature_names, X_test, scaler, y_test


def main():
    st.set_page_config(page_title="Network Anomaly Detection", layout="wide")
    st.title("Real-Time Network Anomaly Detection")
    st.markdown("**Autoencoder** vs **Isolation Forest** • Live scores • Severity • Explainability (AE)")

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "simulator" not in st.session_state:
        st.session_state.simulator = None
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "Both"

    try:
        detector_ae, detector_if, feature_names, X_test, scaler, y_test = get_models_and_data()
    except Exception as e:
        st.error(f"Failed to load model/data: {e}")
        return

    # Sidebar: model selector
    with st.sidebar:
        st.header("Controls")
        model_choice = st.radio(
            "Model",
            ["Autoencoder", "Isolation Forest", "Both"],
            index=2,
            help="Compare both or use a single model.",
        )
        st.session_state.model_choice = model_choice
        auto_refresh = st.checkbox("Auto-refresh (every 2s)", value=False)
        max_display = st.slider("Max packets in table", 10, DASHBOARD_MAX_PACKETS_TABLE, 50)
        if st.button("Clear history"):
            st.session_state.batch_results = []
            st.session_state.simulator = None
            st.rerun()

    use_ae = model_choice in ("Autoencoder", "Both")
    use_if = model_choice in ("Isolation Forest", "Both")
    primary = detector_ae if use_ae else detector_if
    secondary = detector_if if (use_ae and use_if) else None

    if st.session_state.simulator is None:
        st.session_state.simulator = StreamSimulator(
            X_test, primary, batch_size=1, delay_seconds=0, detector_2=secondary
        )
    sim = st.session_state.simulator

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Next packet"):
            try:
                r = next(sim)
                st.session_state.batch_results.append(r)
            except StopIteration:
                st.session_state.simulator = StreamSimulator(
                    X_test, primary, batch_size=1, delay_seconds=0, detector_2=secondary
                )
                r = next(st.session_state.simulator)
                st.session_state.batch_results.append(r)
            st.rerun()
    with col2:
        results = st.session_state.batch_results
        total_ae = sum(int(np.any(res.get("is_anomaly", []))) for res in results)
        total_if = sum(int(np.any(res.get("is_anomaly_2", []))) for res in results)
        if model_choice == "Both":
            st.metric("Anomalies (AE / IF)", f"{total_ae} / {total_if}")
        elif model_choice == "Autoencoder":
            st.metric("Anomalies (AE)", total_ae)
        else:
            st.metric("Anomalies (IF)", total_if)
    with col3:
        st.metric("Total packets", len(results))

    if not results:
        st.info("Click **Next packet** to simulate streaming and compare Autoencoder vs Isolation Forest.")
    else:
        all_scores_ae = []
        all_scores_if = []
        for r in results:
            all_scores_ae.extend(r["scores"])
            if "scores_2" in r:
                all_scores_if.extend(r["scores_2"])

        st.subheader("Anomaly score over time")
        fig = go.Figure()
        if use_ae:
            fig.add_trace(
                go.Scatter(
                    y=all_scores_ae,
                    mode="lines+markers",
                    name="Autoencoder (MSE)",
                    line=dict(color="royalblue"),
                )
            )
        if use_if and all_scores_if:
            fig.add_trace(
                go.Scatter(
                    y=all_scores_if,
                    mode="lines+markers",
                    name="Isolation Forest",
                    line=dict(color="coral"),
                )
            )
        fig.update_layout(
            xaxis_title="Packet index",
            yaxis_title="Anomaly score",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent packets")
        rows = []
        idx = 0
        for r in reversed(results[-max_display * 2 :]):
            for i in range(len(r["scores"])):
                row = {"Index": idx}
                if model_choice == "Both":
                    row["Score (AE)"] = round(r["scores"][i], 6)
                    row["Score (IF)"] = round(r["scores_2"][i], 6) if "scores_2" in r else ""
                else:
                    row["Score"] = round(r["scores"][i], 6)
                row["Severity"] = r["severities"][i]
                row["Anomaly"] = "Yes" if r["is_anomaly"][i] else "No"
                if model_choice == "Both" and "is_anomaly_2" in r:
                    row["Anomaly (IF)"] = "Yes" if r["is_anomaly_2"][i] else "No"
                rows.append(row)
                idx += 1
        if rows:
            df_table = pd.DataFrame(rows[-max_display:])
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        # XAI only for Autoencoder
        st.subheader("Explainability (Autoencoder): top contributing features")
        last_anom_x = None
        if use_ae:
            for r in reversed(results):
                for i, is_anom in enumerate(r["is_anomaly"]):
                    if is_anom:
                        last_anom_x = r["batch"][i]
                        break
                if last_anom_x is not None:
                    break
        if last_anom_x is not None and use_ae:
            top = top_contributing_features(detector_ae.model, last_anom_x, feature_names, top_k=8)
            top_df = pd.DataFrame(top, columns=["Feature", "Reconstruction error"])
            st.dataframe(top_df, use_container_width=True, hide_index=True)
        elif model_choice == "Isolation Forest":
            st.caption("Explainability (feature-level) is available for the Autoencoder model. Switch to Autoencoder or Both to see top contributing features.")
        else:
            st.caption("No anomalous packet (by Autoencoder) in current history to explain.")

    # Evaluation metrics & ROC (when labels available)
    if y_test is not None:
        with st.expander("Evaluation metrics (test set)", expanded=False):
            y_true = (np.array(y_test) != NORMAL_LABEL).astype(int)
            scores_ae = detector_ae.predict_scores(X_test)
            th_ae = detector_ae.threshold or 0.0
            y_pred_ae = (scores_ae > th_ae).astype(int)
            m_ae = compute_metrics(y_true, y_pred_ae)
            roc_ae = compute_roc(y_true, scores_ae)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision (AE)", f"{m_ae['precision']:.3f}")
            c2.metric("Recall (AE)", f"{m_ae['recall']:.3f}")
            c3.metric("F1 (AE)", f"{m_ae['f1']:.3f}")
            c4.metric("ROC AUC (AE)", f"{roc_ae['auc']:.3f}")
            if detector_if is not None:
                scores_if = detector_if.predict_scores(X_test)
                th_if = detector_if.threshold or 0.0
                y_pred_if = (scores_if > th_if).astype(int)
                m_if = compute_metrics(y_true, y_pred_if)
                roc_if = compute_roc(y_true, scores_if)
                st.caption("Isolation Forest:")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precision (IF)", f"{m_if['precision']:.3f}")
                c2.metric("Recall (IF)", f"{m_if['recall']:.3f}")
                c3.metric("F1 (IF)", f"{m_if['f1']:.3f}")
                c4.metric("ROC AUC (IF)", f"{roc_if['auc']:.3f}")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_ae["fpr"], y=roc_ae["tpr"], name=f"AE (AUC={roc_ae['auc']:.3f})", mode="lines"))
            if detector_if is not None:
                fig_roc.add_trace(go.Scatter(x=roc_if["fpr"], y=roc_if["tpr"], name=f"IF (AUC={roc_if['auc']:.3f})", mode="lines"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash")))
            fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", title="ROC Curve", height=350)
            st.plotly_chart(fig_roc, use_container_width=True)

    # Stored anomalies from DB
    try:
        from db.store import get_recent_anomalies
        stored = get_recent_anomalies(limit=50)
        if stored:
            with st.expander("Stored anomalies (database)", expanded=False):
                st.dataframe(pd.DataFrame(stored), use_container_width=True, hide_index=True)
    except Exception:
        pass

    if auto_refresh:
        time.sleep(DASHBOARD_REFRESH_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
