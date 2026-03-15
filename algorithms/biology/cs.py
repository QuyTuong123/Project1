"""Cuckoo Search (CS) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer


class CS(BaseOptimizer):
    """Cuckoo Search optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Number of nests.
        max_iter: Number of iterations.
        pa: Discovery probability (nest abandonment rate).
    """

    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100, pa=0.25):

        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.dim = dim
        self.pa = pa

    def initialize(self):
        """Initialize nests and current best."""

        self.population = np.random.uniform(
            self.lb, self.ub, (self.pop_size, self.dim)
        )
        self.fitness = np.array([
            self.obj_func(ind) for ind in self.population
        ])

        best_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()

    def levy(self):
        """Generate a simplified Levy-like random step."""
        return np.random.randn(self.dim)

    def update(self):
        """Update nests by random walks and abandonment."""
        lb, ub = self.bounds
        for i in range(self.pop_size):
            step = self.levy()
            candidate = self.population[i] + 0.01 * step
            candidate = np.clip(candidate, lb, ub)
            fit = self.obj_func(candidate)
            if fit < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = fit

        abandon = np.random.rand(self.pop_size) < self.pa
        for i in range(self.pop_size):
            if abandon[i]:
                self.population[i] = np.random.uniform(lb, ub, self.dim)
                self.fitness[i] = self.obj_func(self.population[i])

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_score:
            self.gbest = self.population[best_idx].copy()
            self.best_score = self.fitness[best_idx]
            self.best_position = self.gbest.copy()

    def run(self):
        """Execute optimization loop.

        Returns:
            tuple[np.ndarray, float]: Best position and best objective value.
        """
        self.initialize()
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.best_score)
        return self.best_position, self.best_score