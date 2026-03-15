import numpy as np

from core.base_optimizer import BaseOptimizer


class HillClimbing(BaseOptimizer):
    """Steepest-improvement stochastic hill climbing optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Unused placeholder for API compatibility.
        max_iter: Number of iterations.
        step_size: Std of Gaussian neighbor perturbation.
        n_neighbors: Number of neighbors sampled per iteration.
    """

    def __init__(
        self,
        obj_func,
        bounds,
        dim,
        pop_size=1,
        max_iter=100,
        step_size=0.1,
        n_neighbors=20,
    ):
        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.step_size = step_size
        self.n_neighbors = n_neighbors

    def initialize(self):
        """Initialize current and best state uniformly within bounds."""
        self.current = np.random.uniform(self.lb, self.ub, self.dim)
        self.current_score = self.obj_func(self.current)
        self.best_position = self.current.copy()
        self.best_score = self.current_score

    def update(self):
        """Sample neighbors, greedily accept improvements, and update best."""
        neighbors = self.current + np.random.normal(
            0.0, self.step_size, (self.n_neighbors, self.dim)
        )
        neighbors = np.clip(neighbors, self.lb, self.ub)
        scores = np.apply_along_axis(self.obj_func, 1, neighbors)

        best_idx = np.argmin(scores)
        candidate = neighbors[best_idx]
        candidate_score = scores[best_idx]

        if candidate_score < self.current_score:
            self.current = candidate
            self.current_score = candidate_score

        if self.current_score < self.best_score:
            self.best_score = self.current_score
            self.best_position = self.current.copy()
