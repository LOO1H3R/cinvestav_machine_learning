import jax
import jax.numpy as jnp
from jax import grad, random
import numpy as np
import pickle

class Model:
    def __init__(self, key=random.PRNGKey(42), learning_rate=0.01, epochs=1000):
        self.key = key
        self.lr = learning_rate
        self.epochs = epochs
        self.params = None
        self.loss_history = []
        self.acc_history = []

    def fit(self, X, y, X_val=None, y_val=None, print_every=100):
        X = jnp.array(X)
        y = jnp.array(y)
        if X_val is not None and y_val is not None:
             X_val = jnp.array(X_val)
             y_val = jnp.array(y_val)

        self.input_dim = X.shape[1]
        self.params = self._init_params(self.input_dim)
        
        for i in range(self.epochs):
            self.params = self._update(self.params, X, y)
            
            if i % print_every == 0:
                loss_val = self._loss(self.params, X, y)
                self.loss_history.append(loss_val)
                y_pred_train = self.predict(X)
                acc_train = jnp.mean(y_pred_train == y)
                self.acc_history.append(acc_train)
                
    def predict_proba(self, X):
        X = jnp.array(X)
        return self._forward(self.params, X)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def _update(self, params, X, y):
        grads = grad(self._loss)(params, X, y)
        return jax.tree_util.tree_map(lambda p, g: p - self.lr * g, params, grads)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'params': self.params, 'input_dim': self.input_dim}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.input_dim = data['input_dim']

class LogisticRegression(Model):
    def _init_params(self, input_dim):
        self.key, k1, k2 = random.split(self.key, 3)
        return {'w': random.normal(k1, (input_dim,)) * 0.01, 'b': jnp.zeros(())}

    def _forward(self, params, X):
        logits = jnp.dot(X, params['w']) + params['b']
        return jax.nn.sigmoid(logits)

    def _loss(self, params, X, y):
        y_pred = self._forward(params, X)
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
