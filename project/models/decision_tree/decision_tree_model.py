import jax
import jax.numpy as jnp
import numpy as np
import pickle

class JaxDecisionTree:
    """A binary classification decision tree built with NumPy and predicted via JAX."""
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
        self.n_classes_ = 2
        self.tree_ = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.asarray(sample_weight)

        self.tree_ = self._build_tree(X, y, sample_weight, 0)
        return self

    def _build_tree(self, X, y, w, depth):
        total_w = np.sum(w)
        # Class probabilities
        prob = np.array([np.sum(w[y == c]) for c in range(self.n_classes_)]) / max(total_w, 1e-9)

        node = {'prob': prob, 'feature': -1, 'threshold': 0.0, 'left': None, 'right': None}

        if (depth >= self.max_depth or 
            len(np.unique(y)) <= 1 or 
            len(y) < self.min_samples_split):
            return node

        best_impurity = np.inf
        n_features = X.shape[1]
        
        for feature in range(n_features):
            values = X[:, feature]
            thresholds = np.unique(values)
            if len(thresholds) > 50:
                thresholds = np.percentile(values, np.linspace(0, 100, 50))

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                wl = np.sum(w[left_mask])
                wr = np.sum(w[right_mask])
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                yl = y[left_mask]
                wl_norm = w[left_mask] / max(wl, 1e-9)
                gini_l = 1.0 - sum((np.sum(wl_norm[yl == c]))**2 for c in range(self.n_classes_))
                
                yr = y[right_mask]
                wr_norm = w[right_mask] / max(wr, 1e-9)
                gini_r = 1.0 - sum((np.sum(wr_norm[yr == c]))**2 for c in range(self.n_classes_))
                
                impurity = (wl * gini_l + wr * gini_r) / total_w
                if impurity < best_impurity:
                    best_impurity = impurity
                    node['feature'] = feature
                    node['threshold'] = threshold
                    node['left_mask'] = left_mask
                    node['right_mask'] = right_mask

        if node['feature'] != -1:
            node['left'] = self._build_tree(X[node['left_mask']], y[node['left_mask']], w[node['left_mask']], depth + 1)
            node['right'] = self._build_tree(X[node['right_mask']], y[node['right_mask']], w[node['right_mask']], depth + 1)
            del node['left_mask']
            del node['right_mask']
        return node

    def _predict_single(self, node, x):
        if node['feature'] == -1:
            return node['prob']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(node['left'], x)
        else:
            return self._predict_single(node['right'], x)

    def predict_proba(self, X):
        X = np.asarray(X)
        probas = np.array([self._predict_single(self.tree_, x) for x in X])
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class DecisionTreeModel:
    """JAX implementation wrapper replacing sklearn DecisionTreeClassifier."""

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ):
        self.model = JaxDecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return jnp.array(self.model.predict(X))

    def predict_proba(self, X):
        return jnp.array(self.model.predict_proba(X))

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(jnp.mean(y_pred == jnp.array(y)))

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self
