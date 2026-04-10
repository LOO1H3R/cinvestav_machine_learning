import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LogisticRegression
from metaflow import FlowSpec, step
import mlflow

class ChurnModelFlow(FlowSpec):
    
    @step
    def start(self):
        print("Starting Churn Modeling Pipeline with MLflow...")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Telco_Churn_Experiment")
        self.next(self.process_data)
        
    @step
    def process_data(self):
        DATA_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
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
        joblib.dump(scaler, 'scaler.pkl')
        self.features_columns = list(X.columns)
        joblib.dump(self.features_columns, 'features.pkl')
        
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
            
            model.save('model.pkl')
            final_loss = model.loss_history[-1]
            final_acc = model.acc_history[-1]
            
            mlflow.log_metric("final_train_loss", float(final_loss))
            mlflow.log_metric("final_train_acc", float(final_acc))
            mlflow.log_artifact('model.pkl')
            mlflow.log_artifact('scaler.pkl')
            
        self.next(self.end)
        
    @step
    def end(self):
        print("Pipeline is complete!")

if __name__ == '__main__':
    ChurnModelFlow()
