import numpy as np
from core.base_optimizer import BaseOptimizer

class ABC(BaseOptimizer):
    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100, limit=50):

        super().__init__(obj_func, bounds, pop_size, max_iter)
        self.dim = dim
        self.limit = limit

    def initialize(self):
        lb, ub = self.bounds
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)
        self.trial = np.zeros(self.pop_size)

        best_idx = np.argmin(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.gbest_score = self.fitness[best_idx]

    def update(self):
        lb, ub = self.bounds

        # Employed bees
        for i in range(self.pop_size):
            k = np.random.randint(self.pop_size)
            phi = np.random.uniform(-1, 1, self.dim)

            candidate = self.population[i] + \
                        phi * (self.population[i] - self.population[k])

            candidate = np.clip(candidate, lb, ub)
            fit = self.obj_func(candidate)

            if fit < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = fit
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        # Scout bees
        for i in range(self.pop_size):
            if self.trial[i] > self.limit:
                self.population[i] = np.random.uniform(lb, ub, self.dim)
                self.fitness[i] = self.obj_func(self.population[i])
                self.trial[i] = 0

        # Update global best
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.gbest_score:
            self.gbest = self.population[best_idx].copy()
            self.gbest_score = self.fitness[best_idx]

    def run(self):
        self.initialize()
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.gbest_score)
        return self.gbest_score