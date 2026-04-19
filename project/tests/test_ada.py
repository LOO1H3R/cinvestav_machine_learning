import numpy as np
from models.decision_tree_model import JaxDecisionTree

X = np.random.rand(10, 2)
y = np.random.randint(0, 2, 10)
w = np.ones(10) / 10

tree = JaxDecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1)
tree.fit(X, y, sample_weight=w)

print("Tree node:", tree.tree_)
print("Tree predictions:", tree.predict(X))
