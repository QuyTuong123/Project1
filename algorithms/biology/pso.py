"""Particle Swarm Optimization (PSO) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimization optimizer.

    Args:
        obj_func: Objective function to minimize.
        bounds: Tuple of lower/upper bounds.
        dim: Search dimension.
        pop_size: Number of particles.
        max_iter: Number of iterations.
        w: Inertia weight.
        c1: Cognitive coefficient.
        c2: Social coefficient.
    """

    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5):
        
        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.diversity_history = []
        self.trajectory = []
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize(self):
        """Initialize particle states, personal bests, and global best."""
        lb, ub = self.bounds
        self.positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.apply_along_axis(
            self.obj_func, 1, self.positions
        )
        best_idx = np.argmin(self.pbest_scores)
        self.best_position = self.pbest_positions[best_idx].copy()
        self.best_score = self.pbest_scores[best_idx]

    def update(self):
        """Update velocity/position and refresh personal/global best."""
        lb, ub = self.bounds
        for i in range(self.pop_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                + self.c2 * r2 * (self.best_position - self.positions[i])
            )
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], lb, ub)
            fitness = self.obj_func(self.positions[i])
            if fitness < self.pbest_scores[i]:
                self.pbest_scores[i] = fitness
                self.pbest_positions[i] = self.positions[i].copy()
        best_idx = np.argmin(self.pbest_scores)
        self.best_position = self.pbest_positions[best_idx].copy()
        self.best_score = self.pbest_scores[best_idx]

    def run(self):
        """Execute optimization loop.

        Returns:
            tuple[np.ndarray, float]: Best position and best objective value.
        """
        self.initialize()
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.best_score)
            diversity = np.std(self.positions)
            self.diversity_history.append(diversity)
            self.trajectory.append(self.positions.copy())
        return self.best_position, self.best_score