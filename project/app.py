from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Dict
from model import LogisticRegression

app = FastAPI(title="Telco Customer Churn Prediction")

# Load model and scaler at startup
BASE_DIR = Path(__file__).resolve().parent


def _load_prediction_model(path: Path):
    custom_model = LogisticRegression()
    try:
        custom_model.load(path)
        return custom_model
    except Exception:
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


def _build_models_registry() -> Dict[str, object]:
    models = {}

    default_model_path = BASE_DIR / "model.pkl"
    if default_model_path.exists():
        models["jax_logistic"] = _load_prediction_model(default_model_path)

    for artifact_path in sorted(BASE_DIR.glob("*.pkl")):
        if artifact_path.name in {"model.pkl", "features.pkl", "scaler.pkl"}:
            continue
        try:
            models[artifact_path.stem] = _load_prediction_model(artifact_path)
        except Exception:
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

# Tuned per-model classification thresholds.
MODEL_THRESHOLDS = {
    "mlp": 0.55,
}

DEFAULT_MODEL_NAME = next(iter(MODELS))
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = joblib.load(BASE_DIR / "features.pkl")

class CustomerData(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
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
    model_options = "".join(
        [
            (
                f'<option value="{name}"'
                + (' selected="selected"' if name == DEFAULT_MODEL_NAME else '')
                + f'>{name}</option>'
            )
            for name in MODELS.keys()
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
            <p>Fill in customer details and run a quick churn prediction using your trained model.</p>
        </section>

        <section class="card">
            <form id="predict-form">
                <div class="field full"><label for="model_name">Model</label><select id="model_name" name="model_name">__MODEL_OPTIONS__</select></div>
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
    selected_model_name = data.model_name if data.model_name in MODELS else DEFAULT_MODEL_NAME
    _refresh_model_if_needed(selected_model_name)
    selected_model = MODELS[selected_model_name]

    # Convert data to dataframe
    payload = data.dict()
    payload.pop("model_name", None)
    df = pd.DataFrame([payload])
    
    # Process just like in training
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
