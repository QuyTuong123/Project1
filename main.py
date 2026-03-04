from problems.continuous_functions import Sphere
# from algorithms.aco import ACO
from algorithms.pso import PSO
# from algorithms.abc import ABC
# from algorithms.fa import FA
# from algorithms.cs import CS
from visualization.plot_convergence import plot_convergence
from experiments.convergence_analysis import run_30_times
from visualization.plot_convergence import plot_diversity
from visualization.plot_3d_surface import plot_surface, plot_particles_on_surface
from problems.continuous_functions import Sphere
from algorithms.abc import ABC

def main():
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
    #mean_pso, std_pso = run_30_times(PSO)
    #mean_abc, std_abc = run_30_times(ABC)
    #mean_fa, std_fa = run_30_times(FA)
    #mean_cs, std_cs = run_30_times(CS)
    #mean_aco, std_aco = run_30_times(ACO)

    problem30d = Sphere(dim=30)
    optimizer = PSO(
    obj_func=problem30d.evaluate,
    bounds=(problem30d.lb, problem30d.ub),
    dim=30,
    pop_size=50,
    max_iter=200
    )
    best_position, best_score = optimizer_abc.run()
    
    plot_diversity(optimizer.diversity_history)

    print("===== PSO Result =====")
    print("Best Fitness:", best_score)
    print("Best Position:", best_position)

    mean, std, results = run_30_times(PSO)

    print("===== 30 Runs Result =====")
    print("Mean Fitness:", mean)
    print("Std:", std)

    # print("\n===== FINAL COMPARISON =====")
    # print("{:<10} {:<15} {:<15}".format("Algorithm", "Mean", "Std"))
    # print("-" * 45)

    # print("{:<10} {:<15.6e} {:<15.6e}".format("PSO", mean_pso, std_pso))
    # print("{:<10} {:<15.6e} {:<15.6e}".format("ABC", mean_abc, std_abc))
    # print("{:<10} {:<15.6e} {:<15.6e}".format("FA", mean_fa, std_fa))
    # print("{:<10} {:<15.6e} {:<15.6e}".format("CS", mean_cs, std_cs))
    # print("{:<10} {:<15.6e} {:<15.6e}".format("ACO", mean_aco, std_aco))
    
if __name__ == "__main__":
    main()