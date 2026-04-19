import numpy as np


class LinearModel:
    """Simple linear model for testing."""
    
    def __init__(self):
        self.weights = np.ones(50)
        
    def fit(self, X, y, sample_weight=None):
        pass
    
    def predict(self, X):
        """Predict using mean of features."""
        return np.mean(X, axis=1).reshape(-1, 1)
