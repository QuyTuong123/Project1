import numpy as np
import time
from problems.sphere import Sphere

def run_30_times(OptimizerClass, runs=30):
    problem = Sphere(dim=30)
    scores = []
    start = time.time()
    for _ in range(runs):
        try:
            optimizer = OptimizerClass(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim = 30,
                pop_size=30,
                max_iter=100
            )
        except TypeError:
            optimizer = OptimizerClass(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                pop_size=30,
                max_iter=100
            )
        _, best_score = optimizer.run()
        scores.append(best_score)

    end = time.time()
    mean = np.mean(scores)
    std = np.std(scores)
    runtime = end - start

    return mean, std, runtime