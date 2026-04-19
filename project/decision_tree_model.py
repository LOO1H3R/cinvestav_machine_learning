import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel:
    """Thin wrapper around sklearn DecisionTreeClassifier for project consistency."""

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred))

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self
