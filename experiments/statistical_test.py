import time
import numpy as np
from problems.sphere import Sphere
# Biology
from algorithms.biology.pso import PSO
from algorithms.biology.abc import ABC
from algorithms.biology.aco import ACO
from algorithms.biology.fa import FA
from algorithms.biology.cs import CS
# Evolution
from algorithms.evolution.ga import GA
from algorithms.evolution.de import DE
# Physics
from algorithms.physics.sa import SA
# Human
from algorithms.human.tlbo import TLBO

def statistic_test():
    dim = 30
    problem = Sphere(dim=dim)
    algorithms = {
        "PSO": PSO,
        "ABC": ABC,
        "ACO": ACO,
        "FA": FA,
        "CS": CS,
        "GA": GA,
        "DE": DE,
        "SA": SA,
        "TLBO": TLBO
    }

    for name, OptimizerClass in algorithms.items():
        results = []
        times = []
        for _ in range(30):
            optimizer = OptimizerClass(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim=problem.dim,
                pop_size=50,
                max_iter=200
            )

            start = time.time()
            best_position, best_score = optimizer.run()
            end = time.time()
            results.append(best_score)
            times.append(end - start)

        mean = np.mean(results)
        std = np.std(results)
        avg_time = np.mean(times)

        print("\nAlgorithm:", name)
        print("Mean:", f"{mean:.2e}")
        print("Std:", f"{std:.2e}")
        print("Time:", f"{avg_time:.4f}", "seconds")
        print("-" * 40)