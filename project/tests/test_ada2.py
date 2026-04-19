import numpy as np
from models.adaboost_model import AdaBoostModel
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
model = AdaBoostModel(n_estimators=5, max_depth=1)
model.fit(X, y)
print("weights:", model.weights_)
preds = model.predict_proba(X)
print("preds:", preds[:5])
