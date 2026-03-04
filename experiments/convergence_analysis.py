from algorithms.aco import ACO
from algorithms.pso import PSO
from problems.continuous_functions import Sphere
import numpy as np
def run_convergence_test():
    problem = Sphere(dim=30)

    pso = PSO(problem, max_iter=200, population_size=50)
    aco = ACO(problem, max_iter=200, population_size=50)

    pso.run()
    aco.run()

    return {
        "PSO": pso.history,
        "ACO": aco.history
    }

def run_30_times(OptimizerClass):
    results = []

    for _ in range(30):
        problem = Sphere(dim=30)

        optimizer = OptimizerClass(
            obj_func=problem.evaluate,
            bounds=(problem.lb, problem.ub),
            dim=problem.dim,
            pop_size=50,
            max_iter=200
        )

        best_position, best_score = optimizer.run()
        results.append(best_score)

    results = np.array(results)

    mean = np.mean(results)
    std = np.std(results)

    return mean, std, results