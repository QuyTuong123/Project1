"""Teaching-Learning-Based Optimization (TLBO) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer


class TLBO(BaseOptimizer):
    """TLBO optimizer."""

    def __init__(self, obj_func, bounds, dim=30, pop_size=30, max_iter=500):
        super().__init__(obj_func, bounds, pop_size, max_iter, dim)
        self.pop_size = pop_size
        self.lb, self.ub = bounds
        self.population = np.random.uniform(
            self.lb, self.ub, (pop_size, dim)
        )
        self.fitness = np.array([
            obj_func(ind) for ind in self.population
        ])
        
    def initialize(self):
        """Initialize learner population and global best."""
        self.population = np.random.uniform(
            self.lb, self.ub, (self.pop_size, self.dim)
        )
        self.fitness = np.array([
            self.obj_func(ind) for ind in self.population
        ])
        best_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()

    def update(self):
        """Run teacher phase and learner phase for one iteration."""
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]
        mean = np.mean(self.population, axis=0)
        Tf = np.random.randint(1, 3)
        for i in range(self.pop_size):
            r = np.random.rand(self.dim)
            new = self.population[i] + r * (teacher - Tf * mean)
            new = np.clip(new, self.lb, self.ub)
            score = self.obj_func(new)
            if score < self.fitness[i]:
                self.population[i] = new
                self.fitness[i] = score

        for i in range(self.pop_size):
            j = np.random.randint(self.pop_size)
            while j == i:
                j = np.random.randint(self.pop_size)
            Xi = self.population[i]
            Xj = self.population[j]
            r = np.random.rand(self.dim)
            if self.fitness[i] < self.fitness[j]:
                new = Xi + r * (Xi - Xj)
            else:
                new = Xi + r * (Xj - Xi)
            new = np.clip(new, self.lb, self.ub)
            score = self.obj_func(new)
            if score < self.fitness[i]:
                self.population[i] = new
                self.fitness[i] = score

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_score:
            self.best_score = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
