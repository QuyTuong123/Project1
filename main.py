import time
import os
import numpy as np
from problems.sphere import Sphere
from problems.tsp import TSPProblem
# Biology algorithms
from algorithms.biology.abc import ABC
from algorithms.biology.pso import PSO
from algorithms.biology.aco import ACO
from algorithms.biology.fa import FA
from algorithms.biology.cs import CS
# Evolution algorithms
from algorithms.evolution.ga import GA
from algorithms.evolution.de import DE
# Physics algorithms
from algorithms.physics.sa import SA
# Human algorithms
from algorithms.human.tlbo import TLBO
# Visualization
from experiments.convergence_analysis import run_30_times
from visualization.plot_convergence import plot_convergence, plot_diversity
from visualization.plot_3d_surface import plot_surface, plot_particles_on_surface

def main():
    print("\n===== STEP 1: COMPARE CONTINUOUS ALGORITHMS =====")
    problem2d = Sphere(dim=2)
    algorithms = {
        "PSO": PSO,   
        "ABC": ABC,
        "FA": FA,
        "CS": CS,
        "GA": GA,
        "DE": DE,
        "SA": SA,
        "TLBO": TLBO
    }
    results = {}
    for name, Algo in algorithms.items():
        optimizer = Algo(
            obj_func=problem2d.evaluate,
            bounds=(problem2d.lb, problem2d.ub),
            dim=2,
            pop_size=30,
            max_iter=100
        )
        best_position, best_score = optimizer.run()
        results[name] = (best_score, best_position)
    print("\n===== TABLE 1: BEST FITNESS (DIM=2) =====")
    print("{:<10} {:<18} {}".format("Algorithm", "Best Fitness", "Best Position"))
    print("-"*60)

    for name, (fit, pos) in results.items():
        fit_str = f"{fit:.2e}"
        pos_str = "["
        for i, x in enumerate(pos):
            if abs(x) < 1e-4:
                val = f"{x:.1e}"
            else:
                val = f"{x:.5f}"

            if i < len(pos) - 1:
                pos_str += val + " , "
            else:
                pos_str += val
        pos_str += "]"
        print("{:<10} {:<18} {}".format(name, fit_str, pos_str))

    print("\n===== STEP 2: VISUALIZATION (4 HISTOGRAMS) =====")
    optimizer = ABC(
        obj_func=problem2d.evaluate,
        bounds=(problem2d.lb, problem2d.ub),
        dim=2,
        pop_size=30,
        max_iter=100
    )
    optimizer.run()
    # 1 Convergence
    plot_convergence(optimizer.history)
    # 2 Diversity
    plot_diversity(optimizer.diversity_history)
    # 3 Surface
    plot_surface(problem2d.evaluate, problem2d.lb, problem2d.ub)
    # particle movement
    plot_particles_on_surface(problem2d.evaluate, optimizer.trajectory)
    print("\n===== STEP 3: STATISTIC COMPARISION =====")
    algos_8 = {
        "FA": FA,
        "ABC": ABC,
        "CS": CS,
        "PSO": PSO,
        "GA": GA,
        "DE": DE,
        "SA": SA,
        "TLBO": TLBO
    }
    print("\n===== TABLE 2: MEAN / STD / TIME =====")
    print("{:<10} {:<15} {:<15} {:<10}".format(
        "Algorithm", "Mean", "Std", "Time(s)"
    ))

    print("-"*55)
    for name, Algo in algos_8.items():
        start = time.time()
        mean, std, runtime = run_30_times(Algo)
        end = time.time()
        runtime = end - start
        print("{:<10} {:<15.6e} {:<15.6e} {:<10.4f}".format(
        name, mean, std, runtime
        ))
    print("\nDone!")

    print("\n===== STEP 4: TSP WITH ACO (DISCRETE OPTIMIZATION) =====")
    aco = ACO(
        n_cities=20,
        n_ants=10,
        max_iter=50
    )
    best_tour, best_distance = aco.run()
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)

if __name__ == "__main__":
    main()