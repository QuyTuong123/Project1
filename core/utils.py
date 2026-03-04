import numpy as np
def initialize_population(pop_size, dim, bounds):
    lb, ub = bounds
    return np.random.uniform(lb, ub, (pop_size, dim))


def evaluate_population(population, obj_func):
    return np.apply_along_axis(obj_func, 1, population)


def compute_diversity(population):
    return np.std(population)