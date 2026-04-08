# Real-Time Network Anomaly Detection System

Unsupervised real-time network anomaly detection using **Autoencoder** (TensorFlow/Keras) and **Isolation Forest** (scikit-learn), with dynamic thresholding, streaming simulation, **Streamlit** dashboard, **Explainable AI (XAI)**, **Optuna** AutoML, **REST API**, **alerts**, **SQLite storage**, and **Docker** deployment.

## Features

- **Dual models**: Autoencoder (reconstruction MSE) and Isolation Forest; compare in dashboard or CLI
- **Unsupervised detection**: Train on normal traffic only; score = anomaly indicator
- **Real-time streaming**: Packet-by-packet or batch simulation with optional delay
- **Dashboard**: Model selector, live score graph, packet table, severity, XAI, **evaluation metrics (Precision/Recall/F1/ROC)** when labels exist, **stored anomalies** from DB
- **Evaluation**: `evaluate.py` and `main.py --evaluate` for Precision, Recall, F1, ROC curve and AUC (when dataset has labels)
- **Alerts**: Optional email for high-severity anomalies (env: `ALERT_EMAIL_TO`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`); `main.py --alert`
- **Database**: SQLite storage for detected anomalies (`data/anomalies.db`); `main.py --save-db`; view in dashboard expander
- **REST API**: FastAPI app — `POST /detect`, `GET /anomalies`, `GET /models`, `GET /health`; run with `uvicorn api.app:app`
- **Docker**: `Dockerfile` and `docker-compose.yml` for API and dashboard services
- **Modular layout**: `models/`, `utils/`, `streaming/`, `dashboard/`, `xai/`, `tuning/`, `alerts/`, `db/`, `api/`

## Setup

```bash
cd network-cursor
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Data

- Place your network traffic CSV at `data/network_traffic.csv`, or
- Generate sample data: `python -m data.generate_sample_data` or `python main.py --generate-data`

Numeric features; optional label column (`label` / `Class`) with normal=0 for evaluation metrics and ROC.

## Usage

**Train models:**
```bash
python main.py --train        # Train Autoencoder + Isolation Forest
python main.py --train-if     # Train only Isolation Forest
python main.py --tune         # Optuna tuning for Autoencoder, then train
```

**Streaming (with optional DB save and alerts):**
```bash
python main.py --stream 100
python main.py --stream 100 --both --save-db --alert
python main.py --stream 50 --no-delay
```

**Evaluation (when labels available):**
```bash
python main.py --evaluate
python evaluate.py --model both   # Precision, Recall, F1, ROC AUC; saves ROC plot to logs/roc_curve.png if matplotlib present
```

**Dashboard:**
```bash
streamlit run dashboard/app.py
```

**API:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# POST /detect  body: {"features": [ ... ], "model": "autoencoder"}
# GET /anomalies?limit=100
# GET /models  GET /health
```

**Docker:**
```bash
docker-compose up --build
# API: http://localhost:8000  Dashboard: http://localhost:8501
```

**Alerts (optional):** Set env vars `ALERT_EMAIL_TO`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`; use `main.py --alert` when streaming so high-severity anomalies trigger an email.

## Project structure

```
network-cursor/
├── config.py              # Paths, model config, threshold, DB, API port
├── main.py                # Train, tune, stream, --evaluate, --save-db, --alert
├── evaluate.py            # Metrics + ROC (standalone)
├── detector.py            # AnomalyDetector, IsolationForestDetector
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── preprocess.py
│   ├── generate_sample_data.py
│   └── anomalies.db       # Created at runtime if --save-db or API used
├── models/
│   ├── autoencoder.py
│   └── isolation_forest.py
├── utils/
│   ├── threshold.py
│   └── metrics.py         # Precision, Recall, F1, ROC
├── streaming/
│   └── simulator.py
├── xai/
│   └── explain.py
├── tuning/
│   └── optuna_tune.py
├── alerts/
│   └── notify.py          # Email alerts (env-configured)
├── db/
│   └── store.py           # SQLite anomaly storage
├── api/
│   └── app.py             # FastAPI: /detect, /anomalies, /models, /health
├── dashboard/
│   └── app.py             # Streamlit + metrics/ROC + stored anomalies
└── logs/
```
