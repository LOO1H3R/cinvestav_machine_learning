from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import pickle
import sys

import sys
from models import model, adaboost_model, decision_tree_model, linear_model, mlp

# Alias modules so pickle can find them under the old top-level names
sys.modules['model'] = model
sys.modules['adaboost_model'] = adaboost_model
sys.modules['decision_tree_model'] = decision_tree_model
sys.modules['linear_model'] = linear_model
sys.modules['mlp'] = mlp

import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

def accuracy_score(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))

def precision_score(y_true, y_pred, zero_division=0):
    t_p = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    p_p = np.sum(np.array(y_pred) == 1)
    return float(t_p / p_p) if p_p > 0 else float(zero_division)

def recall_score(y_true, y_pred, zero_division=0):
    t_p = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    a_p = np.sum(np.array(y_true) == 1)
    return float(t_p / a_p) if a_p > 0 else float(zero_division)

def f1_score(y_true, y_pred, zero_division=0):
    p = precision_score(y_true, y_pred, zero_division)
    r = recall_score(y_true, y_pred, zero_division)
    return float(2 * (p * r) / (p + r)) if p + r > 0 else float(zero_division)

def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tn), int(fp), int(fn), int(tp)

def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    np.random.seed(random_state)
    y_arr = np.asarray(y)
    t_idx, ts_idx = [], []
    for c in np.unique(y_arr):
        idx = np.where(y_arr == c)[0]
        np.random.shuffle(idx)
        spl = int(len(idx) * (1 - test_size))
        t_idx.extend(idx[:spl])
        ts_idx.extend(idx[spl:])
    if hasattr(X, 'iloc'):
        return X.iloc[t_idx], X.iloc[ts_idx], y[t_idx], y[ts_idx]
    return X[t_idx], X[ts_idx], y[t_idx], y[ts_idx]

try:
    from models.model import LogisticRegression
except ImportError:
    from models.model import LogisticRegression

app = FastAPI(title="Telco Customer Churn Prediction")

# Load model and scaler at startup
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_prediction_model(path: Path):
    name = path.stem
    if "adaboost" in name:
        from models.adaboost_model import AdaBoostModel
        model = AdaBoostModel()
    elif "decision_tree" in name:
        from models.decision_tree_model import DecisionTreeModel
        model = DecisionTreeModel()
    else:
        model = LogisticRegression()
        
    try:
        model.load(path)
        return model
    except (FileNotFoundError, EOFError, KeyError, ValueError, TypeError, AttributeError, pickle.UnpicklingError):
        return joblib.load(path)


def _sigmoid(x: float) -> float:
    """Convert raw logits to probability via sigmoid."""
    return 1.0 / (1.0 + np.exp(-float(x)))


def _extract_churn_probability(raw_proba) -> float:
    proba = np.asarray(raw_proba)
    if proba.ndim == 1:
        return float(proba[0])
    if proba.ndim == 2:
        if proba.shape[1] == 1:
            return float(proba[0, 0])
        return float(proba[0, 1])
    raise ValueError("Unsupported probability output shape")


def _build_models_registry() -> Dict[str, Any]:
    models = {}

    default_model_path = BASE_DIR / "model.pkl"
    if default_model_path.exists():
        models["jax_logistic"] = _load_prediction_model(default_model_path)

    for artifact_path in sorted(BASE_DIR.glob("*.pkl")):
        if artifact_path.name in {"model.pkl", "features.pkl", "scaler.pkl"}:
            continue
        try:
            models[artifact_path.stem] = _load_prediction_model(artifact_path)
        except (FileNotFoundError, EOFError, ValueError, TypeError, AttributeError, pickle.UnpicklingError):
            continue

    return models


def _build_model_paths() -> Dict[str, Path]:
    paths = {}

    default_model_path = BASE_DIR / "model.pkl"
    if default_model_path.exists():
        paths["jax_logistic"] = default_model_path

    for artifact_path in sorted(BASE_DIR.glob("*.pkl")):
        if artifact_path.name in {"model.pkl", "features.pkl", "scaler.pkl"}:
            continue
        paths[artifact_path.stem] = artifact_path

    return paths


def _is_adaboost_variant(model_name: str) -> bool:
    return model_name.endswith("_adaboost")


def _base_model_names() -> list[str]:
    return [
        name
        for name in MODELS.keys()
        if not _is_adaboost_variant(name) and name != "adaboost"
    ]


def _prepare_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    categorical_cols = list(df.select_dtypes(include=["object"]).columns)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[columns]


def _predict_probability(selected_model: Any, x_scaled: np.ndarray) -> float:
    if hasattr(selected_model, "predict_proba"):
        return _extract_churn_probability(selected_model.predict_proba(x_scaled))
    if hasattr(selected_model, "predict"):
        raw_pred = selected_model.predict(x_scaled)
        raw_score = float(np.asarray(raw_pred).flatten()[0])
        return _sigmoid(raw_score)
    raise ValueError("Selected model has neither predict_proba nor predict")


def _predict_probabilities(selected_model: Any, x_scaled: np.ndarray) -> np.ndarray:
    if hasattr(selected_model, "predict_proba"):
        raw = np.asarray(selected_model.predict_proba(x_scaled))
        if raw.ndim == 1:
            return raw.astype(float)
        if raw.ndim == 2:
            if raw.shape[1] == 1:
                return raw[:, 0].astype(float)
            return raw[:, 1].astype(float)
        raise ValueError("Unsupported predict_proba output shape")

    if hasattr(selected_model, "predict"):
        raw_pred = np.asarray(selected_model.predict(x_scaled)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-raw_pred.astype(float)))

    raise ValueError("Selected model has neither predict_proba nor predict")


def _model_threshold(model_name: str) -> float:
    if model_name in MODEL_THRESHOLDS:
        return float(MODEL_THRESHOLDS[model_name])
    if _is_adaboost_variant(model_name):
        base_name = model_name[: -len("_adaboost")]
        return float(MODEL_THRESHOLDS.get(base_name, MODEL_THRESHOLDS.get("adaboost", 0.5)))
    return 0.5


def _build_holdout_split() -> tuple[np.ndarray, np.ndarray]:
    data_file = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not data_file.exists():
        raise FileNotFoundError("Dataset file not found for performance comparison")

    df = pd.read_csv(data_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.dropna()

    y = (df["Churn"] == "Yes").astype(int).values
    x_raw = df.drop(columns=["customerID", "Churn"])
    x_all = _prepare_features_frame(x_raw)

    _, x_test, _, y_test = train_test_split(
        x_all,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    x_test_scaled = scaler.transform(x_test)
    return np.asarray(x_test_scaled), np.asarray(y_test)


def _evaluate_model_on_holdout(model_name: str, model_obj: Any, x_test_scaled: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    threshold = _model_threshold(model_name)
    probs = _predict_probabilities(model_obj, x_test_scaled)
    preds = (probs >= threshold).astype(int)

    return {
        "model": model_name,
        "base_model": model_name[: -len("_adaboost")] if _is_adaboost_variant(model_name) else model_name,
        "variant": "AdaBoost" if _is_adaboost_variant(model_name) else "Base",
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "cm": confusion_matrix(y_test, preds),
    }


def _performance_signature() -> tuple[Any, ...]:
    data_file = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    scaler_file = BASE_DIR / "scaler.pkl"
    features_file = BASE_DIR / "features.pkl"

    model_state = tuple((name, MODEL_MTIMES.get(name)) for name in sorted(MODELS.keys()))
    threshold_state = tuple(sorted((k, float(v)) for k, v in MODEL_THRESHOLDS.items()))
    file_state = (
        data_file.stat().st_mtime if data_file.exists() else None,
        scaler_file.stat().st_mtime if scaler_file.exists() else None,
        features_file.stat().st_mtime if features_file.exists() else None,
    )
    return model_state + threshold_state + file_state


def _load_mlflow_summary() -> list[dict[str, Any]]:
    import sqlite3

    db_path = BASE_DIR / "mlflow.db"
    if not db_path.exists():
        return []

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT
            r.run_uuid AS run_id,
            r.status,
            r.start_time,
            r.end_time,
            r.experiment_id,
            MAX(CASE WHEN m.key = 'test_acc' THEN m.value END) AS test_acc,
            MAX(CASE WHEN m.key = 'train_acc' THEN m.value END) AS train_acc,
            MAX(CASE WHEN t.key = 'mlflow.runName' THEN t.value END) AS run_name
        FROM runs r
        LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
        LEFT JOIN tags t ON r.run_uuid = t.run_uuid
        GROUP BY r.run_uuid
        ORDER BY r.start_time DESC
        """
    )
    rows = [dict(row) for row in cursor.fetchall()]
    connection.close()
    return rows


def _load_metaflow_summary() -> list[dict[str, Any]]:
    flow_root = BASE_DIR / ".metaflow" / "ChurnModelFlow"
    if not flow_root.exists():
        return []

    summary: list[dict[str, Any]] = []
    for run_dir in sorted(flow_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not run_dir.is_dir() or len(run_dir.name) != 16 or not run_dir.name.isdigit() is False:
            # The datastore contains hashed directories and run ids; keep it permissive.
            pass
        metadata = run_dir / "metadata"
        if run_dir.is_dir():
            summary.append(
                {
                    "name": run_dir.name,
                    "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                    "has_metadata": metadata.exists(),
                    "has_artifacts": any((run_dir / child).exists() for child in ["artifacts", "events", "tasks"]),
                }
            )
    return summary[:30]


def _refresh_model_if_needed(model_name: str) -> None:
    path = MODEL_PATHS.get(model_name)
    if path is None or (not path.exists()):
        return

    current_mtime = path.stat().st_mtime
    last_mtime = MODEL_MTIMES.get(model_name)
    if last_mtime is None or current_mtime > last_mtime:
        MODELS[model_name] = _load_prediction_model(path)
        MODEL_MTIMES[model_name] = current_mtime


def _sync_model_registry() -> None:
    """Discover new/updated/removed model artifacts at runtime."""
    latest_paths = _build_model_paths()

    # Remove models that no longer exist.
    for name in list(MODELS.keys()):
        if name not in latest_paths:
            MODELS.pop(name, None)
            MODEL_MTIMES.pop(name, None)

    # Add/update models present on disk.
    for name, path in latest_paths.items():
        MODEL_PATHS[name] = path
        current_mtime = path.stat().st_mtime
        last_mtime = MODEL_MTIMES.get(name)
        if name not in MODELS or last_mtime is None or current_mtime > last_mtime:
            MODELS[name] = _load_prediction_model(path)
            MODEL_MTIMES[name] = current_mtime


MODELS = _build_models_registry()
if not MODELS:
    raise RuntimeError("No loadable models were found in project artifacts")

MODEL_PATHS = _build_model_paths()
MODEL_MTIMES = {name: path.stat().st_mtime for name, path in MODEL_PATHS.items() if path.exists()}
PERFORMANCE_CACHE: dict[str, Any] = {
    "signature": None,
    "rows": None,
}

# Tuned per-model classification thresholds.
MODEL_THRESHOLDS = {
    "mlp": 0.55,
    "decision_tree": 0.50,
    "adaboost": 0.50,
}

DEFAULT_MODEL_NAME = next(iter(_base_model_names()), next(iter(MODELS)))
class JaxStandardScaler:
    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_
        
    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

# Workaround to fix unpickling JaxStandardScaler
sys.modules['__main__'].JaxStandardScaler = JaxStandardScaler

scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = joblib.load(BASE_DIR / "features.pkl")

class CustomerData(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    use_adaboost: bool = False
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    _sync_model_registry()
    base_models = [name for name in _base_model_names() if name in MODEL_PATHS]
    model_options = "".join(
        [
            (
                f'<option value="{name}"'
                + (' selected="selected"' if name == DEFAULT_MODEL_NAME else '')
                + f'>{name}</option>'
            )
            for name in sorted(base_models)
        ]
    )

    page = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Telco Churn Predictor</title>
    <style>
        :root {
            --bg-1: #f6f7fb;
            --bg-2: #e8efff;
            --surface: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0ea5a4;
            --accent-2: #0369a1;
            --danger: #b91c1c;
            --ok: #166534;
            --border: #d1d5db;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
            --radius: 14px;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: "Poppins", "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 10% 10%, #dbeafe 0%, transparent 35%),
                radial-gradient(circle at 90% 20%, #ccfbf1 0%, transparent 30%),
                linear-gradient(160deg, var(--bg-1), var(--bg-2));
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
        }

        .hero {
            background: linear-gradient(135deg, #0ea5a4, #0369a1);
            color: #ffffff;
            border-radius: var(--radius);
            padding: 24px;
            box-shadow: var(--shadow);
            animation: rise 500ms ease-out;
        }

        .hero h1 {
            margin: 0 0 6px 0;
            font-size: clamp(1.4rem, 2.2vw, 2rem);
        }

        .hero p {
            margin: 0;
            opacity: 0.95;
        }

        .card {
            background: var(--surface);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 20px;
            animation: rise 650ms ease-out;
        }

        form {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .field.full { grid-column: 1 / -1; }

        label {
            font-size: 0.9rem;
            color: var(--muted);
            font-weight: 600;
        }

        input, select {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 0.95rem;
            background: #ffffff;
            color: var(--text);
            transition: border-color 150ms ease, box-shadow 150ms ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-2);
            box-shadow: 0 0 0 4px rgba(3, 105, 161, 0.15);
        }

        .actions {
            grid-column: 1 / -1;
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 2px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            border-radius: 999px;
            padding: 11px 18px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            background: linear-gradient(120deg, var(--accent), var(--accent-2));
            color: #ffffff;
            transition: transform 120ms ease, filter 120ms ease;
        }

        button:hover { filter: brightness(1.05); }
        button:active { transform: translateY(1px); }

        .ghost {
            background: #f3f4f6;
            color: var(--text);
            border: 1px solid var(--border);
        }

        .link-btn {
            display: inline-block;
            text-decoration: none;
            border-radius: 999px;
            padding: 11px 18px;
            font-size: 0.95rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: #ffffff;
            color: var(--text);
        }

        .result {
            margin-top: 12px;
            border-radius: 12px;
            padding: 14px;
            border: 1px solid var(--border);
            background: #f9fafb;
            min-height: 56px;
            display: flex;
            align-items: center;
            font-size: 0.98rem;
        }

        .result.ok {
            color: var(--ok);
            border-color: #86efac;
            background: #f0fdf4;
        }

        .result.danger {
            color: var(--danger);
            border-color: #fca5a5;
            background: #fef2f2;
        }

        @media (max-width: 840px) {
            body { padding: 14px; }
            form { grid-template-columns: 1fr; }
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <main class="container">
        <section class="hero">
            <h1>Telco Customer Churn Predictor</h1>
            <p>Fill in customer details and run a quick churn prediction.</p>
        </section>

        <section class="card">
            <form id="predict-form">
                <div class="field full"><label for="model_name">Model</label><select id="model_name" name="model_name">__MODEL_OPTIONS__</select></div>
                <div class="field full"><label for="use_adaboost">Boosting</label><select id="use_adaboost" name="use_adaboost"><option value="false">Base model</option><option value="true">AdaBoost version</option></select></div>
                <div class="field"><label for="gender">Gender</label><select id="gender" name="gender"><option>Female</option><option>Male</option></select></div>
                <div class="field"><label for="SeniorCitizen">Senior Citizen</label><select id="SeniorCitizen" name="SeniorCitizen"><option value="0">0 - No</option><option value="1">1 - Yes</option></select></div>
                <div class="field"><label for="Partner">Partner</label><select id="Partner" name="Partner"><option>Yes</option><option>No</option></select></div>
                <div class="field"><label for="Dependents">Dependents</label><select id="Dependents" name="Dependents"><option>Yes</option><option>No</option></select></div>
                <div class="field"><label for="tenure">Tenure (months)</label><input id="tenure" name="tenure" type="number" min="0" value="12" required /></div>
                <div class="field"><label for="PhoneService">Phone Service</label><select id="PhoneService" name="PhoneService"><option>Yes</option><option>No</option></select></div>
                <div class="field"><label for="MultipleLines">Multiple Lines</label><select id="MultipleLines" name="MultipleLines"><option>No</option><option>Yes</option><option>No phone service</option></select></div>
                <div class="field"><label for="InternetService">Internet Service</label><select id="InternetService" name="InternetService"><option>DSL</option><option>Fiber optic</option><option>No</option></select></div>
                <div class="field"><label for="OnlineSecurity">Online Security</label><select id="OnlineSecurity" name="OnlineSecurity"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="OnlineBackup">Online Backup</label><select id="OnlineBackup" name="OnlineBackup"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="DeviceProtection">Device Protection</label><select id="DeviceProtection" name="DeviceProtection"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="TechSupport">Tech Support</label><select id="TechSupport" name="TechSupport"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="StreamingTV">Streaming TV</label><select id="StreamingTV" name="StreamingTV"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="StreamingMovies">Streaming Movies</label><select id="StreamingMovies" name="StreamingMovies"><option>Yes</option><option>No</option><option>No internet service</option></select></div>
                <div class="field"><label for="Contract">Contract</label><select id="Contract" name="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
                <div class="field"><label for="PaperlessBilling">Paperless Billing</label><select id="PaperlessBilling" name="PaperlessBilling"><option>Yes</option><option>No</option></select></div>
                <div class="field"><label for="PaymentMethod">Payment Method</label><select id="PaymentMethod" name="PaymentMethod"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></div>
                <div class="field"><label for="MonthlyCharges">Monthly Charges</label><input id="MonthlyCharges" name="MonthlyCharges" type="number" min="0" step="0.01" value="70.35" required /></div>
                <div class="field"><label for="TotalCharges">Total Charges</label><input id="TotalCharges" name="TotalCharges" type="number" min="0" step="0.01" value="845.50" required /></div>

                <div class="actions">
                    <button type="submit">Predict Churn</button>
                    <button class="ghost" type="reset">Reset</button>
                    <a class="link-btn" href="/performance">View Model Performance</a>
                    <a class="link-btn" href="/tracking">View Training Data</a>
                </div>
            </form>

            <div id="result" class="result">Submit the form to see prediction results.</div>
        </section>
    </main>

    <script>
        const form = document.getElementById("predict-form");
        const resultBox = document.getElementById("result");

        function toPayload(formData) {
            return {
                model_name: formData.get("model_name"),
                use_adaboost: formData.get("use_adaboost") === "true",
                gender: formData.get("gender"),
                SeniorCitizen: Number(formData.get("SeniorCitizen")),
                Partner: formData.get("Partner"),
                Dependents: formData.get("Dependents"),
                tenure: Number(formData.get("tenure")),
                PhoneService: formData.get("PhoneService"),
                MultipleLines: formData.get("MultipleLines"),
                InternetService: formData.get("InternetService"),
                OnlineSecurity: formData.get("OnlineSecurity"),
                OnlineBackup: formData.get("OnlineBackup"),
                DeviceProtection: formData.get("DeviceProtection"),
                TechSupport: formData.get("TechSupport"),
                StreamingTV: formData.get("StreamingTV"),
                StreamingMovies: formData.get("StreamingMovies"),
                Contract: formData.get("Contract"),
                PaperlessBilling: formData.get("PaperlessBilling"),
                PaymentMethod: formData.get("PaymentMethod"),
                MonthlyCharges: Number(formData.get("MonthlyCharges")),
                TotalCharges: Number(formData.get("TotalCharges"))
            };
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            resultBox.className = "result";
            resultBox.textContent = "Predicting...";

            const payload = toPayload(new FormData(form));

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const data = await response.json();
                const probabilityPct = (data.churn_probability * 100).toFixed(2);
                const isChurn = Number(data.churn_prediction) === 1;

                resultBox.className = `result ${isChurn ? "danger" : "ok"}`;
                resultBox.textContent = isChurn
                    ? `Prediction: Churn (Risk: ${probabilityPct}%)`
                    : `Prediction: No Churn (Risk: ${probabilityPct}%)`;
            } catch (error) {
                resultBox.className = "result danger";
                resultBox.textContent = `Failed to predict. ${error.message}`;
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(page.replace("__MODEL_OPTIONS__", model_options))

@app.post("/predict")
def predict_churn(data: CustomerData):
    _sync_model_registry()
    base_model_name = data.model_name if data.model_name in MODELS else DEFAULT_MODEL_NAME
    if _is_adaboost_variant(base_model_name):
        base_model_name = base_model_name[: -len("_adaboost")]

    selected_model_name = f"{base_model_name}_adaboost" if data.use_adaboost else base_model_name
    if selected_model_name not in MODELS:
        available_boosted = f"{base_model_name}_adaboost" in MODELS
        raise ValueError(
            f"Requested model '{selected_model_name}' not available. "
            f"AdaBoost available for base model '{base_model_name}': {available_boosted}"
        )

    _refresh_model_if_needed(selected_model_name)
    selected_model = MODELS[selected_model_name]

    # Convert data to dataframe
    payload = data.dict()
    payload.pop("model_name", None)
    payload.pop("use_adaboost", None)
    df = pd.DataFrame([payload])
    
    # Process just like in training
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all columns from training match
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Keep only the right columns in the right order
    df_encoded = df_encoded[columns]
    
    # Scale
    x_scaled = scaler.transform(df_encoded)
    
    # Predict
    if hasattr(selected_model, "predict_proba"):
        # Model has predict_proba (e.g., LogisticRegression)
        prob = _extract_churn_probability(selected_model.predict_proba(x_scaled))
    elif hasattr(selected_model, "predict"):
        # Model only has predict (e.g., LinearRegression)—convert output via sigmoid
        raw_pred = selected_model.predict(x_scaled)
        raw_score = float(np.asarray(raw_pred).flatten()[0])
        prob = _sigmoid(raw_score)
    else:
        raise ValueError(f"Selected model '{selected_model_name}' has neither predict_proba nor predict")

    threshold = MODEL_THRESHOLDS.get(selected_model_name, 0.5)
    prediction = int(prob >= threshold)
    
    return {
        "model_used": selected_model_name,
        "threshold_used": float(threshold),
        "churn_prediction": prediction,
        "churn_probability": float(prob)
    }


@app.get("/performance")
def performance_page():
    _sync_model_registry()

    try:
        signature = _performance_signature()
        cached_rows = PERFORMANCE_CACHE.get("rows")

        if PERFORMANCE_CACHE.get("signature") == signature and cached_rows is not None:
            rows = list(cached_rows)
        else:
            x_test_scaled, y_test = _build_holdout_split()
            rows = []
            for model_name in sorted(MODELS.keys()):
                _refresh_model_if_needed(model_name)
                model_obj = MODELS[model_name]
                rows.append(_evaluate_model_on_holdout(model_name, model_obj, x_test_scaled, y_test))
            rows.sort(key=lambda r: r["f1"], reverse=True)
            PERFORMANCE_CACHE["signature"] = signature
            PERFORMANCE_CACHE["rows"] = list(rows)
    except Exception as exc:
        return HTMLResponse(
            f"""<!doctype html>
<html><body style='font-family:Segoe UI, sans-serif; padding:24px;'>
<h2>Performance Comparison</h2>
<p>Failed to compute metrics: {exc}</p>
<p><a href='/'>Back to predictor</a></p>
</body></html>""",
            status_code=500,
        )

    table_rows = "".join(
        [
            (
                "<tr>"
                f"<td>{row['base_model']}</td>"
                f"<td>{row['variant']}</td>"
                f"<td>{row['threshold']:.2f}</td>"
                f"<td>{row['accuracy']:.4f}</td>"
                f"<td>{row['precision']:.4f}</td>"
                f"<td>{row['recall']:.4f}</td>"
                f"<td>{row['f1']:.4f}</td>"
                "</tr>"
            )
            for row in rows
        ]
    )

    best_row = rows[0] if rows else None
    best_text = (
        f"Best by F1: {best_row['model']} (F1={best_row['f1']:.4f})"
        if best_row
        else "No model rows available"
    )

    cm_html_list = []
    for row in rows:
        tn, fp, fn, tp = row["cm"]
        model_display_name = f"{row['base_model']} ({row['variant']})"
        
        # Plot confusion matrix with matplotlib
        import matplotlib.pyplot as plt
        import io
        import base64
        import numpy as np
        
        cm_arr = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(3, 3))
        cax = ax.matshow(cm_arr, cmap='Blues')
        for (i, j), z in np.ndenumerate(cm_arr):
            color = 'white' if z > (cm_arr.max() / 2) else 'black'
            ax.text(j, i, str(z), ha='center', va='center', color=color)
            
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        ax.xaxis.set_ticks_position('bottom')
        plt.title(f"{model_display_name}", pad=20)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        mat_html = f"""
        <div class="cm-card" style="background:#fff; border:1px solid #d1d5db; border-radius:8px; padding:15px; text-align:center;">
            <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;" />
        </div>
        """
        cm_html_list.append(mat_html)
    cm_html_block = "<div style='display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px;'>" + "".join(cm_html_list) + "</div>"

    chart_labels = [f"{row['base_model']} ({row['variant']})" for row in rows]
    chart_payload = {
        "labels": chart_labels,
        "accuracy": [row["accuracy"] for row in rows],
        "precision": [row["precision"] for row in rows],
        "recall": [row["recall"] for row in rows],
        "f1": [row["f1"] for row in rows],
    }
    chart_json = json.dumps(chart_payload)

    html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Model Performance Comparison</title>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <style>
        :root {{
            --bg-1: #f6f7fb;
            --bg-2: #e8efff;
            --surface: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0ea5a4;
            --accent-2: #0369a1;
            --border: #d1d5db;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
            --radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "Poppins", "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 10% 10%, #dbeafe 0%, transparent 35%),
                radial-gradient(circle at 90% 20%, #ccfbf1 0%, transparent 30%),
                linear-gradient(160deg, var(--bg-1), var(--bg-2));
            min-height: 100vh;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; display: grid; gap: 18px; }}
        .hero {{
            background: linear-gradient(135deg, #0ea5a4, #0369a1);
            color: #fff;
            border-radius: var(--radius);
            padding: 24px;
            box-shadow: var(--shadow);
        }}
        .card {{
            background: var(--surface);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 20px;
            overflow-x: auto;
        }}
        .summary {{ color: var(--muted); margin: 0; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
        .chart-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
        }}
        .chart-title {{ margin: 0 0 8px 0; color: var(--text); font-size: 0.98rem; }}
        .chart-wrap {{ position: relative; height: 330px; }}
        table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
        th {{ background: #f8fafc; font-weight: 700; }}
        .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .btn {{
            display: inline-block;
            text-decoration: none;
            border-radius: 999px;
            padding: 10px 16px;
            border: 1px solid var(--border);
            color: var(--text);
            background: #fff;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <main class='container'>
        <section class='hero'>
            <h1 style='margin:0 0 8px 0;'>Model Performance Comparison</h1>
            <p style='margin:0; opacity:.95;'>Metrics.</p>
        </section>

        <section class='card'>
            <p class='summary'>{best_text}</p>
            <div class='charts'>
                <div class='chart-card'>
                    <p class='chart-title'>Core Classification Metrics (Accuracy / Precision / Recall / F1)</p>
                    <div class='chart-wrap'><canvas id='coreMetricsChart'></canvas></div>
                </div>
            </div>
        </section>

        <section class='card'>
            <table>
                <thead>
                    <tr>
                        <th>Base Model</th>
                        <th>Variant</th>
                        <th>Threshold</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>Confusion Matrices</h2>
            {cm_html_block}
        </section>

        <section class='actions'>
            <a class='btn' href='/'>Back to Predictor</a>
            <a class='btn' href='/performance'>Refresh Metrics</a>
            <a class='btn' href='/dataset'>Dataset Overview</a>
        </section>
    </main>

    <script>
        const perfData = {chart_json};

        const coreCtx = document.getElementById('coreMetricsChart').getContext('2d');
        new Chart(coreCtx, {{
            type: 'bar',
            data: {{
                labels: perfData.labels,
                datasets: [
                    {{ label: 'Accuracy', data: perfData.accuracy, backgroundColor: 'rgba(14, 165, 164, 0.70)' }},
                    {{ label: 'Precision', data: perfData.precision, backgroundColor: 'rgba(3, 105, 161, 0.70)' }},
                    {{ label: 'Recall', data: perfData.recall, backgroundColor: 'rgba(34, 197, 94, 0.70)' }},
                    {{ label: 'F1', data: perfData.f1, backgroundColor: 'rgba(249, 115, 22, 0.70)' }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'Score' }} }},
                    x: {{ ticks: {{ maxRotation: 35, minRotation: 20 }} }}
                }},
                plugins: {{
                    legend: {{ position: 'top' }},
                    tooltip: {{ mode: 'index', intersect: false }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    return HTMLResponse(html)


@app.get("/tracking")
def tracking_page():
        mlflow_rows = _load_mlflow_summary()
        metaflow_rows = _load_metaflow_summary()

        mlflow_table = "".join(
                [
                        (
                                "<tr>"
                                f"<td>{row.get('run_name') or row.get('run_id')}</td>"
                                f"<td>{row.get('status', '')}</td>"
                                f"<td>{row.get('experiment_id', '')}</td>"
                                f"<td>{row.get('train_acc') if row.get('train_acc') is not None else ''}</td>"
                                f"<td>{row.get('test_acc') if row.get('test_acc') is not None else ''}</td>"
                                f"<td>{row.get('start_time') or ''}</td>"
                                "</tr>"
                        )
                        for row in mlflow_rows[:25]
                ]
        ) or "<tr><td colspan='6'>No MLflow runs found.</td></tr>"

        metaflow_table = "".join(
                [
                        (
                                "<tr>"
                                f"<td>{row['name']}</td>"
                                f"<td>{row['modified']}</td>"
                                f"<td>{'yes' if row['has_metadata'] else 'no'}</td>"
                                f"<td>{'yes' if row['has_artifacts'] else 'no'}</td>"
                                "</tr>"
                        )
                        for row in metaflow_rows
                ]
        ) or "<tr><td colspan='4'>No Metaflow datastore entries found.</td></tr>"

        html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Tracking Summary</title>
    <style>
        :root {{
            --bg-1: #f6f7fb;
            --bg-2: #e8efff;
            --surface: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0ea5a4;
            --accent-2: #0369a1;
            --border: #d1d5db;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
            --radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin:0; font-family:"Poppins","Segoe UI",sans-serif; color:var(--text); background: linear-gradient(160deg, var(--bg-1), var(--bg-2)); min-height:100vh; padding:24px; }}
        .container {{ max-width:1200px; margin:0 auto; display:grid; gap:18px; }}
        .hero {{ background: linear-gradient(135deg, #0ea5a4, #0369a1); color:#fff; border-radius:var(--radius); padding:24px; box-shadow:var(--shadow); }}
        .card {{ background:var(--surface); border:1px solid rgba(255,255,255,0.35); border-radius:var(--radius); box-shadow:var(--shadow); padding:20px; overflow-x:auto; }}
        table {{ width:100%; border-collapse:collapse; min-width:900px; }}
        th, td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }}
        th {{ background:#f8fafc; font-weight:700; }}
        .actions {{ display:flex; gap:10px; flex-wrap:wrap; }}
        .btn {{ display:inline-block; text-decoration:none; border-radius:999px; padding:10px 16px; border:1px solid var(--border); color:var(--text); background:#fff; font-weight:600; }}
        .muted {{ color: var(--muted); }}
    </style>
</head>
<body>
    <main class='container'>
        <section class='hero'>
            <h1 style='margin:0 0 8px 0;'>Tracking Summary</h1>
            <p style='margin:0; opacity:.95;'>This page reads the local MLflow database and Metaflow datastore directly from the project folder.</p>
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>MLflow Runs</h2>
            <table>
                <thead>
                    <tr><th>Run</th><th>Status</th><th>Experiment</th><th>Train Acc</th><th>Test Acc</th><th>Started</th></tr>
                </thead>
                <tbody>{mlflow_table}</tbody>
            </table>
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>Metaflow Runs</h2>
            <p class='muted'>Showing local datastore entries under <code>.metaflow/ChurnModelFlow</code>.</p>
            <table>
                <thead>
                    <tr><th>Entry</th><th>Modified</th><th>Metadata</th><th>Artifacts</th></tr>
                </thead>
                <tbody>{metaflow_table}</tbody>
            </table>
        </section>

        <section class='actions'>
            <a class='btn' href='/'>Back to Predictor</a>
            <a class='btn' href='/performance'>Performance Comparison</a>
            <a class='btn' href='/dataset'>Dataset Overview</a>
        </section>
    </main>
</body>
</html>"""

        return HTMLResponse(html)




@app.get("/dataset")
def dataset_page():
    import pandas as pd
    import matplotlib.pyplot as plt
    import io
    import base64
    
    data_file = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not data_file.exists():
        return HTMLResponse("<h1>Dataset not found.</h1>")
    
    df = pd.read_csv(data_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    n_rows, n_cols = df.shape
    missing_vals = df.isnull().sum().sum()
    churn_counts = df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {}
    
    # Generate Plots
    def get_plot_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # Numerical
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for i, col in enumerate(num_cols):
        df_no = df[df["Churn"] == "No"][col].dropna()
        df_yes = df[df["Churn"] == "Yes"][col].dropna()
        axes[i].hist([df_no, df_yes], bins=30, stacked=True, color=['#3b82f6', '#ef4444'], label=['No Churn', 'Churn'], edgecolor='white')
        axes[i].set_title(f'Distribution of {col}')
        if i == 0:
            axes[i].legend()
    plt.tight_layout()
    num_b64 = get_plot_base64()

    # Categorical
    cat_cols = ["gender", "InternetService", "Contract"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df["Churn"])
        for c in ["No", "Yes"]:
            if c not in ct.columns: ct[c] = 0
        ct = ct[["No", "Yes"]] # ensure order
        ct.plot(kind='bar', stacked=True, color=['#3b82f6', '#ef4444'], ax=axes2[i], legend=False)
        axes2[i].set_title(f'Counts of {col}')
        axes2[i].tick_params(axis='x', labelrotation=0)
    fig2.legend(['No Churn', 'Churn'], loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    cat_b64 = get_plot_base64()

    html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Dataset Overview</title>
    <style>
        :root {{
            --bg-1: #f6f7fb; --bg-2: #e8efff; --surface: #ffffff;
            --text: #1f2937; --muted: #6b7280; --border: #d1d5db;
            --shadow: 0 12px 30px rgba(15,23,42,0.12); --radius: 14px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; font-family: "Poppins", "Segoe UI", sans-serif;
            color: var(--text); padding: 24px;
            background: linear-gradient(160deg, var(--bg-1), var(--bg-2));
            min-height: 100vh;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; display: grid; gap: 18px; }}
        .hero {{
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            color: #fff; border-radius: var(--radius); padding: 24px; box-shadow: var(--shadow);
        }}
        .card {{
            background: var(--surface); border: 1px solid rgba(255,255,255,0.35);
            border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px; overflow-x: auto;
        }}
        img {{ max-width: 100%; height: auto; display: block; border-radius: 8px; }}
        .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 5px; }}
        .btn {{
            display: inline-block; text-decoration: none; border-radius: 999px;
            padding: 10px 16px; border: 1px solid var(--border); color: var(--text);
            background: #fff; font-weight: 600; font-size: 0.95rem;
        }}
        .btn:hover {{ background: #f1f5f9; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
        .stat-box {{
            padding: 15px; background: #f8fafc; border-radius: 12px;
            text-align: center; border: 1px solid var(--border);
        }}
        .stat-value {{ font-size: 1.8em; font-weight: 700; color: #8b5cf6; margin-top: 8px; }}
        .stat-label {{ color: var(--muted); font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
    </style>
</head>
<body>
    <main class='container'>
        <section class='hero'>
            <h1 style='margin:0 0 8px 0;'>Dataset Visual Overview</h1>
            <p style='margin:0; opacity:.95;'>Key feature distributions graphically displayed.</p>
        </section>

        <section class='card'>
            <div class='stat-grid'>
                <div class='stat-box'><div class='stat-label'>Total Rows</div><div class='stat-value'>{n_rows}</div></div>
                <div class='stat-box'><div class='stat-label'>Total Columns</div><div class='stat-value'>{n_cols}</div></div>
                <div class='stat-box'><div class='stat-label'>Missing Values</div><div class='stat-value'>{missing_vals}</div></div>
                <div class='stat-box'><div class='stat-label'>Churn (Yes)</div><div class='stat-value'>{churn_counts.get('Yes', 0)}</div></div>
                <div class='stat-box'><div class='stat-label'>Churn (No)</div><div class='stat-value'>{churn_counts.get('No', 0)}</div></div>
            </div>
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>Numerical Feature Distributions</h2>
            <img src='data:image/png;base64,{num_b64}' alt='Numerical Distributions' />
        </section>

        <section class='card'>
            <h2 style='margin-top:0;'>Categorical Summaries</h2>
            <img src='data:image/png;base64,{cat_b64}' alt='Categorical Counts' />
        </section>

        <section class='actions'>
            <a class='btn' href='/'>Back to Predictor</a>
            <a class='btn' href='/performance'>Performance Comparison</a>
            <a class='btn' href='/dataset'>Dataset Overview</a>
        </section>
    </main>
</body>
</html>"""

    return HTMLResponse(html)

if __name__ == "__main__":



    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
