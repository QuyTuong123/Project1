import numpy as np
from core.base_optimizer import BaseOptimizer

class GA(BaseOptimizer):
    def initialize(self):
        lb, ub = self.bounds
        self.population = np.random.uniform(
            lb, ub, (self.pop_size, self.dim)
        )

        self.fitness = np.apply_along_axis(
            self.obj_func, 1, self.population
        )
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_score = self.fitness[best_idx]
        self.best_position = self.population[self.best_idx].copy()
        
    def selection(self):
        idx1, idx2 = np.random.choice(self.pop_size,2)
        if self.fitness[idx1] < self.fitness[idx2]:
            return self.population[idx1]
        return self.population[idx2]
    
    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        child = alpha*parent1 + (1-alpha)*parent2
        return child
    
    def mutation(self, child):
        mutation_rate = 0.1
        if np.random.rand() < mutation_rate:
            noise = np.random.normal(0,0.1,len(child))
            child = child + noise
        child = np.clip(child, self.lb, self.ub)
        return child
    
    def update(self):
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
            self.best_solution = self.population[best_idx]
            self.history.append(self.best_score)
