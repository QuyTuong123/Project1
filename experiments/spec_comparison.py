import time

from algorithms.classical.astar import astar_search
from algorithms.classical.bfs import bfs_search
from algorithms.classical.dfs import dfs_search
from algorithms.classical.greedy import greedy_best_first_search
from algorithms.classical.ucs import ucs_search
from algorithms.physics.hill_climbing import HillClimbing
from algorithms.physics.sa import SA
from problems.grid_pathfinding import GridPathfindingProblem
from problems.sphere import Sphere


def run_classical_search_comparison(log_func=print):
    """Compare BFS/DFS/UCS/Greedy/A* on a common grid-pathfinding benchmark."""
    problem = GridPathfindingProblem(width=25, height=25, obstacle_ratio=0.22, seed=42)

    algorithms = {
        "BFS": bfs_search,
        "DFS": dfs_search,
        "UCS": ucs_search,
        "Greedy": greedy_best_first_search,
        "A*": astar_search,
    }

    rows = []
    for name, algo in algorithms.items():
        t0 = time.perf_counter()
        out = algo(problem)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "Algorithm": name,
                "Found": "Yes" if out["found"] else "No",
                "PathLen": int(out["cost"]) if out["found"] else -1,
                "Expanded": int(out["expanded"]),
                "TimeMs": dt * 1000.0,
            }
        )

    log_func("\n===== SPEC COMPARISON A: CLASSICAL SEARCH =====")
    log_func("{:<10} {:<8} {:<10} {:<10} {:<10}".format("Algorithm", "Found", "PathLen", "Expanded", "Time(ms)"))
    log_func("-" * 56)
    for r in rows:
        path_len_str = str(r["PathLen"]) if r["PathLen"] >= 0 else "-"
        log_func(
            "{:<10} {:<8} {:<10} {:<10} {:<10.3f}".format(
                r["Algorithm"],
                r["Found"],
                path_len_str,
                r["Expanded"],
                r["TimeMs"],
            )
        )

    return rows


def run_local_search_comparison(log_func=print, runs=30):
    """Compare Hill Climbing and Simulated Annealing on Sphere benchmark."""
    problem = Sphere(dim=30)
    algorithms = {
        "HillClimb": HillClimbing,
        "SA": SA,
    }

    log_func("\n===== SPEC COMPARISON B: LOCAL SEARCH =====")
    log_func("{:<12} {:<15} {:<15} {:<10}".format("Algorithm", "Mean", "Std", "Time(s)"))
    log_func("-" * 58)

    summary = {}
    for name, Algo in algorithms.items():
        scores = []
        t0 = time.perf_counter()
        for _ in range(runs):
            optimizer = Algo(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim=problem.dim,
                pop_size=30,
                max_iter=200,
            )
            _, best_score = optimizer.run()
            scores.append(best_score)

        dt = time.perf_counter() - t0
        import numpy as np

        mean = float(np.mean(scores))
        std = float(np.std(scores))
        summary[name] = {"mean": mean, "std": std, "time": dt}
        log_func("{:<12} {:<15.6e} {:<15.6e} {:<10.4f}".format(name, mean, std, dt))

    return summary


def run_spec_comparison(log_func=print):
    class_rows = run_classical_search_comparison(log_func=log_func)
    local_rows = run_local_search_comparison(log_func=log_func, runs=30)
    return {"classical": class_rows, "local": local_rows}
