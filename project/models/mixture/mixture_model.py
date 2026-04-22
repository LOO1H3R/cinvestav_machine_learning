import jax
import jax.numpy as jnp
from jax import grad, random
import numpy as np
import pickle

class MixtureModel:
    """Mixture of Experts for binary classification using JAX."""
    
    def __init__(self, key=random.PRNGKey(42), num_experts=2, learning_rate=0.05, epochs=500):
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.key = key
        self.params = None
        self.input_dim = None
        self.loss_history = []

    def _init_params(self, input_dim):
        self.key, k1, k2 = random.split(self.key, 3)
        return {
            'w_gate': random.normal(k1, (input_dim, self.num_experts)) * 0.1,
            'b_gate': jnp.zeros(self.num_experts),
            'w_expert': random.normal(k2, (input_dim, self.num_experts)) * 0.1,
            'b_expert': jnp.zeros(self.num_experts)
        }

    def _forward(self, params, X):
        X = jnp.asarray(X)
        # Gating network
        gate_logits = jnp.dot(X, params['w_gate']) + params['b_gate']
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)  # shape: (N, num_experts)
        
        # Expert networks (each outputs probability of class 1)
        expert_logits = jnp.dot(X, params['w_expert']) + params['b_expert']
        expert_probs = jax.nn.sigmoid(expert_logits)  # shape: (N, num_experts)
        
        # Mixture prediction
        out_prob = jnp.sum(gate_probs * expert_probs, axis=-1)
        return out_prob

    def _loss(self, params, X, y, sample_weight=None):
        y_pred = self._forward(params, X)
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
        if sample_weight is not None:
             return jnp.sum(bce * sample_weight)
        return jnp.mean(bce)

    def _update(self, params, X, y, sample_weight=None):
        grads = grad(self._loss)(params, X, y, sample_weight)
        return jax.tree_util.tree_map(
            lambda p, g: p - self.learning_rate * g,
            params,
            grads
        )

    def fit(self, X, y, X_val=None, y_val=None, print_every=100, sample_weight=None):
        X = jnp.array(X)
        y = jnp.array(y).flatten()

        self.input_dim = X.shape[1]
        self.params = self._init_params(self.input_dim)

        for epoch in range(self.epochs):
            self.params = self._update(self.params, X, y, sample_weight)

            if epoch % print_every == 0:
                loss_val = self._loss(self.params, X, y, sample_weight)
                self.loss_history.append(loss_val)
        return self

    def predict_proba(self, X):
        X = jnp.array(X)
        prob1 = self._forward(self.params, X).reshape(-1, 1)
        prob0 = 1.0 - prob1
        return jnp.hstack((prob0, prob1))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'input_dim': self.input_dim,
                'num_experts': self.num_experts,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.input_dim = data['input_dim']
            self.num_experts = data.get('num_experts', 2)
            self.epochs = data.get('epochs', 500)
            self.learning_rate = data.get('learning_rate', 0.05)
        return self
