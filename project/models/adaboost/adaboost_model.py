import jax.numpy as jnp
import numpy as np
import pickle
try:
    from project.models.decision_tree.decision_tree_model import JaxDecisionTree
except ModuleNotFoundError:
    from models.decision_tree.decision_tree_model import JaxDecisionTree

import copy

class AdaBoostModel:
    """Pure JAX/NumPy AdaBoost classifier replacing sklearn AdaBoostClassifier."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs # handle discarded max_depth from train_flow
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.weights_ = []
        if base_estimator is None:
            self.base_estimator = JaxDecisionTree(max_depth=kwargs.get('max_depth', 2))
        else:
            self.base_estimator = base_estimator

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        y_encoded = np.where(y == 1, 1, -1)
        
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.weights_ = []
        
        for _ in range(self.n_estimators):
            if hasattr(self.base_estimator, 'random_state'):
                tree = copy.deepcopy(self.base_estimator)
                tree.random_state = self.random_state + _
            else:
                tree = copy.deepcopy(self.base_estimator)
            
            # Linear model predicting mean shape mismatch fix
            if hasattr(tree, 'learning_rate') and getattr(tree, 'epochs', 0) == 0:
                pass
            
            tree.fit(X, y, sample_weight=w.reshape(-1, 1) if 'MLP' in tree.__class__.__name__ else w)
            
            
            y_pred = np.array(tree.predict(X)).flatten()
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
            pred_tree = np.array(tree.predict(X)).flatten()
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
