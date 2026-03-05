import numpy as np
from core.base_optimizer import BaseOptimizer
from core.utils import initialize_population
class CS(BaseOptimizer):
    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100, pa=0.25):

        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.dim = dim
        self.pa = pa

    def initialize(self):
        lb, ub = self.bounds
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)

        best_idx = np.argmin(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.best_score = self.fitness[best_idx]

    def levy(self):
        return np.random.randn(self.dim)

    def update(self):
        lb, ub = self.bounds

        # Generate new solutions
        for i in range(self.pop_size):
            step = self.levy()
            candidate = self.population[i] + 0.01 * step
            candidate = np.clip(candidate, lb, ub)

            fit = self.obj_func(candidate)

            if fit < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = fit

        # Abandon some nests
        abandon = np.random.rand(self.pop_size) < self.pa
        for i in range(self.pop_size):
            if abandon[i]:
                self.population[i] = np.random.uniform(lb, ub, self.dim)
                self.fitness[i] = self.obj_func(self.population[i])

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_score:
            self.gbest = self.population[best_idx].copy()
            self.best_score = self.fitness[best_idx]

    def run(self):
        self.initialize()
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.best_score)
        return self.gbest, self.best_score