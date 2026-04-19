import re

# app.py patches
with open("app.py", "r") as f:
    app_code = f.read()

app_code = app_code.replace("from .model import LogisticRegression", "from project.models.model import LogisticRegression")
app_code = app_code.replace("from model import LogisticRegression", "from project.models.model import LogisticRegression")
app_code = app_code.replace("from project.adaboost_model import AdaBoostModel", "from project.models.adaboost_model import AdaBoostModel")
app_code = app_code.replace("from project.decision_tree_model import DecisionTreeModel", "from project.models.decision_tree_model import DecisionTreeModel")
app_code = app_code.replace('BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"', 'BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"')

with open("app.py", "w") as f:
    f.write(app_code)

# train_flow.py patches
with open("train_flow.py", "r") as f:
    train_code = f.read()

train_code = train_code.replace("from model import LogisticRegression", "from models.model import LogisticRegression")
train_code = train_code.replace("from decision_tree_model import DecisionTreeModel", "from models.decision_tree_model import DecisionTreeModel")
train_code = train_code.replace("from adaboost_model import AdaBoostModel", "from models.adaboost_model import AdaBoostModel")
train_code = train_code.replace("self.base_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'", "self.base_dir / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")

with open("train_flow.py", "w") as f:
    f.write(train_code)

print("Restructure patches applied")
