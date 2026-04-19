import jax
import jax.numpy as jnp
from jax import grad
import pickle
import numpy as np

class LinearModel:
    """Logistic regression model using JAX."""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.input_dim = None

    def _sigmoid(self, z):
        return jax.nn.sigmoid(z)

    def _loss(self, params, X, y, sample_weight=None):
        w, b = params
        z = jnp.dot(X, w) + b
        y_pred = jnp.clip(self._sigmoid(z), 1e-7, 1 - 1e-7)
        bce = -(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
        if sample_weight is not None:
             return jnp.sum(bce * sample_weight)
        return jnp.mean(bce)

    def fit(self, X, y, sample_weight=None):
        X = jnp.array(X)
        y = jnp.array(y).reshape(-1)
        if sample_weight is not None:
             sample_weight = jnp.array(sample_weight).reshape(-1)

        self.input_dim = X.shape[1]
        
        # Initialize parameters
        rng = np.random.default_rng(42)
        w = jnp.array(rng.normal(0, 0.01, self.input_dim))
        b = jnp.array(0.0)
        
        params = (w, b)
        loss_grad_fn = grad(self._loss)

        for epoch in range(self.epochs):
            grads = loss_grad_fn(params, X, y, sample_weight)
            w = w - self.learning_rate * grads[0]
            b = b - self.learning_rate * grads[1]
            params = (w, b)
        
        self.weights, self.bias = params
        return self

    def predict_proba(self, X):
        X = jnp.array(X)
        z = jnp.dot(X, self.weights) + self.bias
        prob1 = self._sigmoid(z).reshape(-1, 1)
        proba = jnp.hstack((1 - prob1, prob1))
        return proba

    def predict(self, X):
        X = jnp.array(X)
        z = jnp.dot(X, self.weights) + self.bias
        return (z > 0).astype(int).reshape(-1)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'input_dim': self.input_dim,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = jnp.array(data['weights'])
            self.bias = jnp.array(data['bias'])
            self.input_dim = data['input_dim']
            self.learning_rate = data.get('learning_rate', 0.01)
            self.epochs = data.get('epochs', 1000)
        return self
