import jax
import jax.numpy as jnp
from jax import grad, random
import numpy as np


class MLPClassifier:
    """Multilayer Perceptron for binary classification using JAX."""
    
    def __init__(self, hidden_dims=[64, 32], key=random.PRNGKey(42), learning_rate=0.01, epochs=100):
        self.hidden_dims = hidden_dims
        self.key = key
        self.lr = learning_rate
        self.epochs = epochs
        self.params = None
        self.input_dim = None
        self.loss_history = []
        self.acc_history = []
    
    def _init_params(self, input_dim):
        """Initialize network parameters."""
        key = self.key
        dims = [input_dim] + self.hidden_dims + [1]
        params = {'w': [], 'b': []}
        
        for i in range(len(dims) - 1):
            key, k1, k2 = random.split(key, 3)
            scale = jnp.sqrt(2.0 / dims[i])
            params['w'].append(random.normal(k1, (dims[i], dims[i+1])) * scale)
            params['b'].append(jnp.zeros(dims[i+1]))
        
        return params
    
    def _forward(self, params, X):
        """Forward pass through the network."""
        z = X
        # Hidden layers with ReLU
        for i in range(len(self.hidden_dims)):
            z = jnp.dot(z, params['w'][i]) + params['b'][i]
            z = jnp.maximum(z, 0)  # ReLU
        
        # Output layer with sigmoid
        logits = jnp.dot(z, params['w'][-1]) + params['b'][-1]
        return jax.nn.sigmoid(logits)
    
    def _loss(self, params, X, y):
        """Binary cross-entropy loss."""
        y_pred = self._forward(params, X)
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
    
    def _update(self, params, X, y):
        """Update parameters using gradient descent."""
        grads = grad(self._loss)(params, X, y)
        return jax.tree_util.tree_map(
            lambda p, g: p - self.lr * g,
            params,
            grads
        )
    
    def fit(self, X, y, X_val=None, y_val=None, print_every=100):
        """Train the network."""
        X = jnp.array(X)
        y = jnp.array(y).reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        self.params = self._init_params(self.input_dim)
        
        for epoch in range(self.epochs):
            self.params = self._update(self.params, X, y)
            
            if epoch % print_every == 0:
                loss_val = self._loss(self.params, X, y)
                self.loss_history.append(loss_val)
    
    def predict(self, X):
        """Get raw predictions (probabilities)."""
        X = jnp.array(X)
        return jnp.asarray(self._forward(self.params, X))
    
    def predict_proba(self, X):
        """Get class probabilities."""
        return self.predict(X)
    
    def save(self, path):
        """Save model parameters."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'hidden_dims': self.hidden_dims,
                'input_dim': self.input_dim
            }, f)
    
    def load(self, path):
        """Load model parameters."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.hidden_dims = data['hidden_dims']
            self.input_dim = data['input_dim']
