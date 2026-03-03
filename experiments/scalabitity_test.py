import time
from problems.continuous_functions import Sphere
from algorithms.pso import PSO

def test_scalability():

    dimensions = [10, 50, 100]

    for dim in dimensions:
        problem = Sphere(dim=dim)

        optimizer = PSO(
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

        print(f"Dimension: {dim}")
        print(f"Best Fitness: {best_score:.2e}")
        print(f"Runtime: {runtime:.4f} seconds")
        print("-" * 30)