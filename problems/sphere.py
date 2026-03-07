import numpy as np

class Sphere:
    def __init__(self, dim=30):
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)
        self.global_optimum = 0

    def evaluate(self, x):
        x = np.array(x)
        return np.sum(x**2)