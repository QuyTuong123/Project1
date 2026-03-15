"""Firefly Algorithm (FA) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer


class FA(BaseOptimizer):
    """Firefly Algorithm optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Number of fireflies.
        max_iter: Number of iterations.
        alpha: Randomization factor.
        beta0: Base attractiveness.
        gamma: Light absorption coefficient.
    """

    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100,
                 alpha=0.2, beta0=1, gamma=1):
        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def initialize(self):
        """Initialize firefly population and current best."""
        lb, ub = self.bounds
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.population)
        best_idx = np.argmin(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.best_score = self.fitness[best_idx]
        self.best_position = self.gbest.copy()

    def update(self):
        """Move fireflies toward brighter ones and update best."""
        lb, ub = self.bounds
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if self.fitness[j] < self.fitness[i]:
                    r = np.linalg.norm(self.population[i] - self.population[j])
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    self.population[i] += \
                        beta * (self.population[j] - self.population[i]) + \
                        self.alpha * np.random.randn(self.dim)
                    self.population[i] = np.clip(self.population[i], lb, ub)
                    self.fitness[i] = self.obj_func(self.population[i])
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
        self.history = []
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.best_score)
        return self.best_position, self.best_score