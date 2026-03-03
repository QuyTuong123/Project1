from problems.continuous_functions import Sphere
from algorithms.pso import PSO
from visualization.plot_convergence import plot_convergence
from experiments.convergence_analysis import run_30_times
from experiments.scalability_test import test_scalability
from visualization.plot_convergence import plot_diversity
from visualization.plot_3d_surface import plot_surface, plot_particles_on_surface
from problems.continuous_functions import Sphere
from algorithms.abc import ABC
import os
print("Đang chạy tại:", os.getcwd())

def main():

    # ===== ABC 2D DEMO =====
    problem2d = Sphere(dim=2)

    optimizer_abc = ABC(
        obj_func=problem2d.evaluate,
        bounds=(problem2d.lb, problem2d.ub),
        dim=2,
        pop_size=20,
        max_iter=50
    )

    optimizer_abc.run()

    plot_surface(problem2d.evaluate, problem2d.lb, problem2d.ub)
    plot_particles_on_surface(problem2d.evaluate, optimizer_abc.trajectory)

    # ===== PSO 30D =====
    problem = Sphere(dim=30)

    optimizer = PSO(
        obj_func=problem.evaluate,
        bounds=(problem.lb, problem.ub),
        dim=problem.dim,
        pop_size=50,
        max_iter=200
    )

    best_position, best_score = optimizer.run()

    plot_diversity(optimizer.diversity_history)

    print("===== PSO Result =====")
    print("Best Fitness:", best_score)
    print("Best Position:", best_position)

    mean, std, results = run_30_times()

    print("===== 30 Runs Result =====")
    print("Mean Fitness:", mean)
    print("Std:", std)

    test_scalability()


if __name__ == "__main__":
    main()