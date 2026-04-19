import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LogisticRegression
from decision_tree_model import DecisionTreeModel
from adaboost_model import AdaBoostModel
from metaflow import FlowSpec, step
import mlflow

class ChurnModelFlow(FlowSpec):
    
    @step
    def start(self):
        print("Starting Churn Modeling Pipeline with MLflow...")
        self.base_dir = Path(__file__).resolve().parent
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features_columns = None
        mlflow_db = (self.base_dir / "mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        mlflow.set_experiment("Telco_Churn_Experiment")
        self.next(self.process_data)
        
    @step
    def process_data(self):
        DATA_FILE = self.base_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        df = pd.read_csv(DATA_FILE)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
        df.dropna(inplace=True)
        
        df_clean = df.drop('customerID', axis=1)
        df_clean['Churn'] = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        # Save scaler for API usage
        import joblib
        joblib.dump(scaler, str(self.base_dir / 'scaler.pkl'))
        self.features_columns = list(X.columns)
        joblib.dump(self.features_columns, str(self.base_dir / 'features.pkl'))
        
        self.next(self.train_model)
        
    @step
    def train_model(self):
        mlflow.set_experiment("Telco_Churn_Experiment")

        with mlflow.start_run(run_name="Logistic_Regression"):
            epochs = 1000
            lr = 0.1
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            
            model = LogisticRegression(learning_rate=lr, epochs=epochs)
            model.fit(self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test, print_every=100)
            
            model.save(str(self.base_dir / 'model.pkl'))
            final_loss = model.loss_history[-1]
            final_acc = model.acc_history[-1]
            
            mlflow.log_metric("final_train_loss", float(final_loss))
            mlflow.log_metric("final_train_acc", float(final_acc))
            y_test_pred = model.predict(self.X_test)
            test_acc = float((y_test_pred == self.y_test).mean())
            mlflow.log_metric("test_acc", test_acc)
            mlflow.log_artifact(str(self.base_dir / 'model.pkl'))
            mlflow.log_artifact(str(self.base_dir / 'scaler.pkl'))
            mlflow.log_artifact(str(self.base_dir / 'features.pkl'))

        with mlflow.start_run(run_name="Decision_Tree"):
            max_depth = 8
            min_samples_split = 20
            min_samples_leaf = 10
            random_state = 42
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("random_state", random_state)

            tree_model = DecisionTreeModel(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            tree_model.fit(self.X_train, self.y_train)
            tree_train_acc = tree_model.score(self.X_train, self.y_train)
            tree_test_acc = tree_model.score(self.X_test, self.y_test)
            tree_path = self.base_dir / "decision_tree.pkl"
            tree_model.save(str(tree_path))

            mlflow.log_metric("train_acc", tree_train_acc)
            mlflow.log_metric("test_acc", tree_test_acc)
            mlflow.log_artifact(str(tree_path))
            mlflow.log_artifact(str(self.base_dir / 'scaler.pkl'))
            mlflow.log_artifact(str(self.base_dir / 'features.pkl'))

        adaboost_variants = {
            "jax_logistic": {"n_estimators": 260, "learning_rate": 0.06, "max_depth": 2, "random_state": 42},
            "decision_tree": {"n_estimators": 220, "learning_rate": 0.05, "max_depth": 2, "random_state": 42},
            "linear": {"n_estimators": 200, "learning_rate": 0.08, "max_depth": 1, "random_state": 42},
            "mlp": {"n_estimators": 320, "learning_rate": 0.04, "max_depth": 3, "random_state": 42},
        }

        for base_name, cfg in adaboost_variants.items():
            with mlflow.start_run(run_name=f"AdaBoost_{base_name}"):
                mlflow.log_param("base_model", base_name)
                mlflow.log_param("n_estimators", cfg["n_estimators"])
                mlflow.log_param("learning_rate", cfg["learning_rate"])
                mlflow.log_param("max_depth", cfg["max_depth"])
                mlflow.log_param("random_state", cfg["random_state"])

                adaboost_model = AdaBoostModel(
                    n_estimators=cfg["n_estimators"],
                    learning_rate=cfg["learning_rate"],
                    max_depth=cfg["max_depth"],
                    random_state=cfg["random_state"],
                )
                adaboost_model.fit(self.X_train, self.y_train)
                ada_train_acc = adaboost_model.score(self.X_train, self.y_train)
                ada_test_acc = adaboost_model.score(self.X_test, self.y_test)
                artifact_name = f"{base_name}_adaboost.pkl"
                artifact_path = self.base_dir / artifact_name
                adaboost_model.save(str(artifact_path))

                mlflow.log_metric("train_acc", ada_train_acc)
                mlflow.log_metric("test_acc", ada_test_acc)
                mlflow.log_artifact(str(artifact_path))
                mlflow.log_artifact(str(self.base_dir / 'scaler.pkl'))
                mlflow.log_artifact(str(self.base_dir / 'features.pkl'))
            
        self.next(self.end)
        
    @step
    def end(self):
        print("Pipeline is complete! Trained and logged Logistic Regression + Decision Tree + per-model AdaBoost variants.")

if __name__ == '__main__':
    ChurnModelFlow()
