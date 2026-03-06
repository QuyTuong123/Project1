import numpy as np
from core.base_optimizer import BaseOptimizer

class DE(BaseOptimizer):
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
        
    def mutation(self,i):
        idxs = list(range(self.pop_size))
        idxs.remove(i)
        r1,r2,r3 = np.random.choice(idxs,3,replace=False)
        F = 0.8
        mutant = self.population[r1] + F*(self.population[r2]-self.population[r3])
        return mutant
        
    def crossover(self,target,mutant):
        CR = 0.9
        trial = np.copy(target)
        for j in range(len(target)):
            if np.random.rand() < CR:
                trial[j] = mutant[j]
        trial = np.clip(trial, self.lb, self.ub)
        return trial
        
    def update(self):
        for i in range(self.pop_size):
            mutant = self.mutation(i)
            trial = self.crossover(self.population[i], mutant)
            trial_fit = self.obj_func(trial)
            if trial_fit < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fit
                if trial_fit < self.best_score:
                    self.best_score = trial_fit
                    self.best_solution = trial
            

