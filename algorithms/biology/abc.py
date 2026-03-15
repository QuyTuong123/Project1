"""Artificial Bee Colony (ABC) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer


class ABC(BaseOptimizer):
    """Artificial Bee Colony optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Number of bees (solutions).
        max_iter: Number of iterations.
        limit: Abandonment limit before scout reinitialization.
    """

    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100, limit=50):

        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.dim = dim
        self.limit = limit
        self.history = []
        self.trajectory = []
        self.diversity_history = []
    def initialize(self):
        """Initialize population, fitness, and global best."""
        lb, ub = self.bounds
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)
        self.trial = np.zeros(self.pop_size)

        best_idx = np.argmin(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.best_score = self.fitness[best_idx]
        self.best_position = self.gbest.copy()

    def update(self):
        """Run employed/onlooker/scout phases for one iteration."""
        lb, ub = self.bounds
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

        for i in range(self.pop_size):
            if self.trial[i] > self.limit:
                self.population[i] = np.random.uniform(lb, ub, self.dim)
                self.fitness[i] = self.obj_func(self.population[i])
                self.trial[i] = 0

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_score:
            self.gbest = self.population[best_idx].copy()
            self.best_score = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()

    def run(self):
        """Execute optimization loop.

        Returns:
            tuple[np.ndarray, float]: Best position and best objective value.
        """
        self.initialize()
        for _ in range(self.max_iter):
            self.update()

            #convergence
            self.history.append(self.best_score)
            #diversity
            diversity = np.std(self.population)
            self.diversity_history.append(diversity)

            # trajectory
            self.trajectory.append(self.population.copy())
        return self.best_position ,self.best_score