"""Genetic Algorithm (GA) for continuous optimization."""

import numpy as np

from core.base_optimizer import BaseOptimizer

class GA(BaseOptimizer):
    """Genetic Algorithm optimizer."""

    def initialize(self):
        """Initialize population, fitness, and current best."""
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best_score = self.fitness[self.best_idx]
        self.best_position = self.population[self.best_idx].copy()
        
    def selection(self):
        """Tournament selection with size 2."""
        idx1, idx2 = np.random.choice(self.pop_size,2)
        if self.fitness[idx1] < self.fitness[idx2]:
            return self.population[idx1]
        return self.population[idx2]
    
    def crossover(self, parent1, parent2):
        """Arithmetic crossover."""
        alpha = np.random.rand()
        child = alpha*parent1 + (1-alpha)*parent2
        return child
    
    def mutation(self, child):
        """Gaussian mutation with fixed mutation probability."""
        mutation_rate = 0.1
        if np.random.rand() < mutation_rate:
            noise = np.random.normal(0,0.1,len(child))
            child = child + noise
        child = np.clip(child, self.lb, self.ub)
        return child
    
    def update(self):
        """Create new generation and refresh global best."""
        new_population = []
        for _ in range(self.pop_size):
            p1 = self.selection()
            p2 = self.selection()
            child = self.crossover(p1,p2)
            child = self.mutation(child)
            new_population.append(child)
        self.population = np.array(new_population)
        self.fitness = np.array([
            self.obj_func(x) for x in self.population
        ])
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_score:
            self.best_score = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
