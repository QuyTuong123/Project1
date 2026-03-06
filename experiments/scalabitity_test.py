import time
from problems.sphere import Sphere
# Biology
from algorithms.biology.abc import ABC
from algorithms.biology.pso import PSO
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
def test_scalability():
    dimensions = [10, 50, 100]
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
    for dim in dimensions:
        print(f"\n===== Dimension: {dim} =====")
        problem = Sphere(dim=dim)
        for name, OptimizerClass in algorithms.items():
            optimizer = OptimizerClass(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim=problem.dim,
                pop_size=50,
                max_iter=200
            )
            start_time = time.time()
            best_position, best_score = optimizer.run()
            end_time = time.time()
            runtime = end_time - start_time
            print(f"{name}")
            print(f"Best Fitness: {best_score:.2e}")
            print(f"Runtime: {runtime:.4f} seconds")
            print("-" * 30)