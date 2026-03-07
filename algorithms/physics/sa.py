import numpy as np
from core.base_optimizer import BaseOptimizer

class SA(BaseOptimizer):

    def initialize(self):
        dim = self.dim
        low = self.lb
        high = self.ub
        self.current = np.random.uniform(low, high, dim)
        self.current_score = self.obj_func(self.current)
        self.best_position = self.current.copy()
        self.best_score = self.current_score
        self.temperature = 100

    def neighbor(self):
        step = np.random.normal(0, 0.1, self.dim)
        candidate = self.current + step
        candidate = np.clip(candidate, self.lb, self.ub)
        return candidate

    def update(self):
        candidate = self.neighbor()
        score = self.obj_func(candidate)
        delta = score - self.current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.current = candidate
            self.current_score = score
        if score < self.best_score:
            self.best_position = candidate.copy()
            self.best_score = score
        self.temperature *= 0.95