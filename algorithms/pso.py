import numpy as np
from core.base_optimizer import BaseOptimizer

class PSO(BaseOptimizer):
    def __init__(self, obj_func, bounds, dim,
                 pop_size=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5):
        
        super().__init__(obj_func, bounds, pop_size, max_iter)
        self.diversity_history = []
        self.trajectory = []
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def initialize(self):
        lb, ub = self.bounds

        self.positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))

        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))

        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.apply_along_axis(
            self.obj_func, 1, self.positions
        )

        best_idx = np.argmin(self.pbest_scores)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_score = self.pbest_scores[best_idx]

    def update(self):
        lb, ub = self.bounds

        for i in range(self.pop_size):

            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            # Update velocity
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                + self.c2 * r2 * (self.gbest_position - self.positions[i])
            )

            # Update position
            self.positions[i] += self.velocities[i]

            # Clamp vào bounds
            self.positions[i] = np.clip(self.positions[i], lb, ub)

            # Tính fitness mới
            fitness = self.obj_func(self.positions[i])

            # Update pbest
            if fitness < self.pbest_scores[i]:
                self.pbest_scores[i] = fitness
                self.pbest_positions[i] = self.positions[i].copy()

        # Update gbest sau khi cập nhật tất cả particle
        best_idx = np.argmin(self.pbest_scores)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_score = self.pbest_scores[best_idx]


    def run(self):
        self.initialize()

        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.gbest_score)
            diversity = np.std(self.positions)
            self.diversity_history.append(diversity)
            self.trajectory.append(self.positions.copy())
            
        return self.gbest_position, self.gbest_score