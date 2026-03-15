"""Simulated Annealing (SA) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer

class SA(BaseOptimizer):
    """Simulated Annealing optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Unused placeholder for API compatibility.
        max_iter: Number of iterations.
        initial_temperature: Initial annealing temperature.
        cooling_rate: Multiplicative cooling factor.
        step_sigma: Std of Gaussian neighbor perturbation.
    """

    def __init__(
        self,
        obj_func,
        bounds,
        dim,
        pop_size=1,
        max_iter=100,
        initial_temperature=100.0,
        cooling_rate=0.95,
        step_sigma=0.1,
    ):
        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_sigma = step_sigma

    def initialize(self):
        """Initialize current state, best state, and temperature."""
        dim = self.dim
        low = self.lb
        high = self.ub
        self.current = np.random.uniform(low, high, dim)
        self.current_score = self.obj_func(self.current)
        self.best_position = self.current.copy()
        self.best_score = self.current_score
        self.temperature = self.initial_temperature

    def neighbor(self):
        """Sample a clipped Gaussian neighbor around current state."""
        step = np.random.normal(0.0, self.step_sigma, self.dim)
        candidate = self.current + step
        candidate = np.clip(candidate, self.lb, self.ub)
        return candidate

    def update(self):
        """Perform one SA transition and cooling step."""
        candidate = self.neighbor()
        score = self.obj_func(candidate)
        delta = score - self.current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
            self.current = candidate
            self.current_score = score
        if score < self.best_score:
            self.best_position = candidate.copy()
            self.best_score = score
        self.temperature *= self.cooling_rate