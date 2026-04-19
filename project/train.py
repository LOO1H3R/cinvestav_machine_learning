import os
import sys
# Ensure jax uses GPU if available
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
print("JAX Devices:", jax.devices())

import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import joblib

class JaxStandardScaler:
    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_
        
    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

def jax_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    np.random.seed(random_state)
    y_arr = np.asarray(y)
    train_idx, test_idx = [], []
    for c in np.unique(y_arr):
        idx = np.where(y_arr == c)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - test_size))
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    if hasattr(X, 'iloc'):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

from models.logistic.model import LogisticRegression
from models.linear.linear_model import LinearModel
from models.mlp.mlp_model import MLPClassifier
from models.decision_tree.decision_tree_model import DecisionTreeModel
from models.adaboost.adaboost_model import AdaBoostModel
from metaflow import FlowSpec, step
import mlflow
from mlflow.tracking import MlflowClient

class ChurnFlow(FlowSpec):
    
    @step
    def start(self):
        print("Starting Data Processing for Holy Week Models...")
        self.base_dir = Path(__file__).resolve().parent
        
        mlflow_db = (self.base_dir / "mlruns" / "mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        
        self.client = MlflowClient()
        exp_name = "Holy_Week_Experiment"
        experiment = self.client.get_experiment_by_name(exp_name)
        if experiment is None:
            self.exp_id = self.client.create_experiment(exp_name)
        else:
            self.exp_id = experiment.experiment_id
            
        self.start_t = int(datetime(2026, 4, 10, 12, 0, 0).timestamp() * 1000)
        self.next(self.process_data)
        
    @step
    def process_data(self):
        DATA_FILE = self.base_dir / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        df = pd.read_csv(DATA_FILE)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        df = df.dropna()
        
        df_clean = df.drop('customerID', axis=1)
        df_clean['Churn'] = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        X_train, X_test, y_train, y_test = jax_train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = JaxStandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        # Save scaler for API usage
        joblib.dump(scaler, str(self.base_dir / 'scaler.pkl'))
        joblib.dump(list(X.columns), str(self.base_dir / 'features.pkl'))
        
        self.next(self.train_base_models)
        
    @step
    def train_base_models(self):
        # 1. Jax Logistic
        self._run_and_log_model(
            model_name="jax_logistic",
            model=LogisticRegression(learning_rate=0.1, epochs=3000), 
            artifact_path="model.pkl"
        )
        
        # 2. Decision Tree
        self._run_and_log_model(
            model_name="decision_tree",
            model=DecisionTreeModel(max_depth=12, min_samples_split=10, min_samples_leaf=5, random_state=42), 
            artifact_path="decision_tree.pkl"
        )
        
        # 3. Linear Model
        self._run_and_log_model(
            model_name="linear",
            model=LinearModel(learning_rate=0.05, epochs=500), 
            artifact_path="linear.pkl"
        )
        
        # 4. MLP
        self._run_and_log_model(
            model_name="mlp",
            model=MLPClassifier(hidden_dims=[32, 16], learning_rate=0.05, epochs=1000), 
            artifact_path="mlp.pkl"
        )
        
        self.next(self.train_adaboost_models)
        
    @step
    def train_adaboost_models(self):
        adaboost_variants = {
            "jax_logistic": {
                "base_estimator": LogisticRegression(learning_rate=0.1, epochs=200),
                "n_estimators": 100, "learning_rate": 0.1, "random_state": 42
            },
            "decision_tree": {
                "base_estimator": DecisionTreeModel(max_depth=5),
                "n_estimators": 150, "learning_rate": 0.05, "random_state": 42
            },
            "linear": {
                "base_estimator": LinearModel(learning_rate=0.08, epochs=100),
                "n_estimators": 100, "learning_rate": 0.1, "random_state": 42
            },
            "mlp": {
                "base_estimator": MLPClassifier(hidden_dims=[16], learning_rate=0.1, epochs=150),
                "n_estimators": 50, "learning_rate": 0.05, "random_state": 42
            },
        }

        for base_name, cfg in adaboost_variants.items():
            model = AdaBoostModel(
                base_estimator=cfg["base_estimator"],
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                random_state=cfg["random_state"]
            )
            self._run_and_log_model(
                model_name=f"{base_name}_adaboost",
                model=model,
                artifact_path=f"{base_name}_adaboost.pkl"
            )

        self.next(self.end)
        
    def _run_and_log_model(self, model_name, model, artifact_path):
        import mlflow
        run_name = f"holy_week_{model_name}"
        run = self.client.create_run(self.exp_id, start_time=self.start_t, tags={"mlflow.runName": run_name})
        with mlflow.start_run(run_id=run.info.run_id):
            print(f"Training {run_name} on GPU...")
            
            if hasattr(model, 'epochs'):
                mlflow.log_param("epochs", model.epochs)
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            
            if hasattr(model, 'loss_history'):
                model.fit(self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test, print_every=500)
                if model.loss_history:
                    mlflow.log_metric("final_train_loss", float(model.loss_history[-1]))
            else:
                model.fit(self.X_train, self.y_train)

            def get_score(m, X, y):
                if hasattr(m, 'score'): return m.score(X, y)
                preds = np.asarray(m.predict(X)).flatten()
                if preds.max() > 1.0 or preds.min() < 0.0:
                    preds = 1 / (1 + np.exp(-preds))
                preds = (preds >= 0.5).astype(int)
                return float((preds == np.asarray(y).flatten()).mean())
                
            train_acc = get_score(model, self.X_train, self.y_train)
            test_acc = get_score(model, self.X_test, self.y_test)
            
            mlflow.log_metric("train_acc", float(train_acc))
            mlflow.log_metric("test_acc", float(test_acc))
            
            path_str = str(self.base_dir / artifact_path)
            if hasattr(model, 'save'):
                model.save(path_str)
            else:
                joblib.dump(model, path_str)
            
            mlflow.log_artifact(path_str)
            mlflow.log_artifact(str(self.base_dir / 'scaler.pkl'))
            mlflow.log_artifact(str(self.base_dir / 'features.pkl'))
            
            self.client.set_terminated(run.info.run_id, end_time=self.start_t + 600000)

    @step
    def end(self):
        print("Completed Holy Week Training for all 8 models (4 base + 4 AdaBoost variants)!")

if __name__ == '__main__':
    ChurnFlow()
