import numpy as np
from core.base_optimizer import BaseOptimizer

class SA(BaseOptimizer):
    def initialize(self):
        dim = len(self.bounds)
        low = self.lb
        high = self.ub
        self.population = np.random.uniform(
            low, high, (self.pop_size, dim)
        )
        self.fitness = np.array([
            self.obj_func(x) for x in self.population
        ])
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_score = self.fitness[best_idx]
    
    def initialize(self):
        dim = len(self.bounds)
        low = self.lb
        high = self.ub
        self.current = np.random.uniform(low,high,dim)
        self.current_score = self.obj_func(self.current)
        self.best_solution = self.current
        self.best_score = self.current_score
        self.temperature = 100

    def neighbor(self):
        step = np.random.normal(0,0.1,len(self.current))
        return self.current + step
    
    def update(self):
        candidate = self.neighbor()
        score = self.obj_func(candidate)
        delta = score - self.current_score
        candidate = np.clip(candidate, self.lb, self.ub)
        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.current = candidate
            self.current_score = score
            if score < self.best_score:
                self.best = candidate
                self.best_score = score
        self.temperature *= 0.95