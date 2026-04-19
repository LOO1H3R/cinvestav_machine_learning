import jax.numpy as jnp
import numpy as np
import pickle
from decision_tree_model import JaxDecisionTree

class AdaBoostModel:
    """Pure JAX/NumPy AdaBoost classifier replacing sklearn AdaBoostClassifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.weights_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        y_encoded = np.where(y == 1, 1, -1)
        
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.weights_ = []
        
        for _ in range(self.n_estimators):
            tree = JaxDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=2, 
                min_samples_leaf=1,
                random_state=self.random_state
            )
            tree.fit(X, y, sample_weight=w)
            
            y_pred = np.array(tree.predict(X))
            y_pred_encoded = np.where(y_pred == 1, 1, -1)
            
            incorrect = (y_pred_encoded != y_encoded)
            err = np.sum(w * incorrect) / max(np.sum(w), 1e-9)
            
            if err <= 1e-10:
                alpha = 1.0
                self.estimators_.append(tree)
                self.weights_.append(alpha)
                break
                
            err = max(err, 1e-10)
            err = min(err, 1.0 - 1e-10)
            
            alpha = self.learning_rate * np.log((1.0 - err) / err)
            
            w = w * np.exp(alpha * incorrect)
            w = w / np.sum(w)
            
            self.estimators_.append(tree)
            self.weights_.append(alpha)
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        pred_sum = np.zeros(X.shape[0])
        
        for alpha, tree in zip(self.weights_, self.estimators_):
            pred_tree = np.array(tree.predict(X))
            pred_tree_encoded = np.where(pred_tree == 1, 1, -1)
            pred_sum += alpha * pred_tree_encoded
            
        proba_1 = 1.0 / (1.0 + np.exp(-2.0 * pred_sum))
        proba = np.column_stack((1.0 - proba_1, proba_1))
        return jnp.array(proba)

    def predict(self, X):
        return jnp.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(jnp.mean(y_pred == jnp.array(y)))

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'estimators_': self.estimators_, 'weights_': self.weights_}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.estimators_ = data['estimators_']
            self.weights_ = data['weights_']
        return self
