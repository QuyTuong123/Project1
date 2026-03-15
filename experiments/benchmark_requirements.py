import time
import tracemalloc
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from algorithms.biology.abc import ABC
from algorithms.biology.cs import CS
from algorithms.biology.fa import FA
from algorithms.biology.pso import PSO
from algorithms.evolution.de import DE
from algorithms.evolution.ga import GA
from algorithms.human.tlbo import TLBO
from algorithms.physics.hill_climbing import HillClimbing
from algorithms.physics.sa import SA
from problems.continuous_benchmarks import (
    AckleyBenchmark,
    GriewankBenchmark,
    RastriginBenchmark,
    RosenbrockBenchmark,
    SphereBenchmark,
)
from problems.grid_pathfinding import GridPathfindingProblem
from experiments.spec_comparison import run_classical_search_comparison


def _build_optimizer(name, problem, max_iter=100, pop_size=30):
    common = {
        "obj_func": problem.evaluate,
        "bounds": (problem.lb, problem.ub),
        "dim": problem.dim,
        "pop_size": pop_size,
        "max_iter": max_iter,
    }
    factories = {
        "PSO": lambda: PSO(**common),
        "ABC": lambda: ABC(**common),
        "FA": lambda: FA(**common),
        "CS": lambda: CS(**common),
        "GA": lambda: GA(**common),
        "DE": lambda: DE(**common),
        "SA": lambda: SA(**common),
        "HillClimb": lambda: HillClimbing(**common),
        "TLBO": lambda: TLBO(**common),
    }
    return factories[name]()


def _run_single(name, problem, max_iter=100, pop_size=30):
    opt = _build_optimizer(name, problem, max_iter=max_iter, pop_size=pop_size)
    tracemalloc.start()
    t0 = time.perf_counter()
    _, best_score = opt.run()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    hist = np.asarray(getattr(opt, "history", []), dtype=float)
    diversity = np.asarray(getattr(opt, "diversity_history", []), dtype=float)
    return {
        "best": float(best_score),
        "time": float(dt),
        "peak_kb": float(peak / 1024.0),
        "history": hist,
        "diversity": diversity,
    }


def run_continuous_benchmark(runs=10, max_iter=100, pop_size=30):
    algos = ["PSO", "ABC", "FA", "CS", "GA", "DE", "SA", "HillClimb", "TLBO"]
    problems = [
        SphereBenchmark(dim=30),
        RastriginBenchmark(dim=30),
        RosenbrockBenchmark(dim=30),
        GriewankBenchmark(dim=30),
        AckleyBenchmark(dim=30),
    ]

    rows = []
    for pb in problems:
        for algo in algos:
            run_outputs = [
                _run_single(algo, pb, max_iter=max_iter, pop_size=pop_size)
                for _ in range(runs)
            ]
            best_scores = np.array([r["best"] for r in run_outputs], dtype=float)
            times = np.array([r["time"] for r in run_outputs], dtype=float)
            peaks = np.array([r["peak_kb"] for r in run_outputs], dtype=float)

            rows.append(
                {
                    "problem": pb.name,
                    "algorithm": algo,
                    "best": float(np.min(best_scores)),
                    "mean": float(np.mean(best_scores)),
                    "std": float(np.std(best_scores)),
                    "avg_time": float(np.mean(times)),
                    "avg_peak_kb": float(np.mean(peaks)),
                }
            )
    return rows


def run_scalability(runs=5, dims=(10, 30, 50), max_iter=100, pop_size=30):
    algos = ["PSO", "GA", "DE", "SA", "HillClimb"]
    rows = []
    for dim in dims:
        pb = SphereBenchmark(dim=dim)
        for algo in algos:
            scores = []
            times = []
            for _ in range(runs):
                out = _run_single(algo, pb, max_iter=max_iter, pop_size=pop_size)
                scores.append(out["best"])
                times.append(out["time"])
            rows.append(
                {
                    "dim": int(dim),
                    "algorithm": algo,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "avg_time": float(np.mean(times)),
                }
            )
    return rows


def run_exploration_exploitation(max_iter=100, pop_size=30):
    pb = SphereBenchmark(dim=30)
    rows = []
    for algo in ["PSO", "ABC"]:
        out = _run_single(algo, pb, max_iter=max_iter, pop_size=pop_size)
        diversity = out["diversity"]
        if diversity.size > 1:
            start_div = float(diversity[0])
            end_div = float(diversity[-1])
            ratio = float(end_div / (start_div + 1e-12))
        else:
            start_div = 0.0
            end_div = 0.0
            ratio = 0.0
        rows.append(
            {
                "algorithm": algo,
                "start_diversity": start_div,
                "end_diversity": end_div,
                "end_start_ratio": ratio,
            }
        )
    return rows


def save_convergence_plot(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pb = SphereBenchmark(dim=30)
    algorithms = ["PSO", "GA", "DE", "SA", "HillClimb"]

    fig = Figure(figsize=(8, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for algo in algorithms:
        out = _run_single(algo, pb, max_iter=150, pop_size=30)
        if out["history"].size > 0:
            ax.plot(out["history"], label=algo)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Convergence Speed Comparison (Sphere-30D)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "requirements_convergence.png", dpi=180)


def _save_robustness_boxplot(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sphere_rows = [r for r in rows if r["problem"] == "Sphere"]
    names = [r["algorithm"] for r in sphere_rows]
    means = [r["mean"] for r in sphere_rows]
    stds = [r["std"] for r in sphere_rows]

    x = np.arange(len(names))
    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("Mean Best Score (+/- std)")
    ax.set_title("Robustness on Sphere-30D")
    fig.tight_layout()
    fig.savefig(output_dir / "requirements_robustness.png", dpi=180)


def _write_table(rows, file_path):
    if not rows:
        Path(file_path).write_text("\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                vals.append(f"{v:.6e}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_requirement_benchmarks(
    output_dir="visualizes",
    table_dir="experiments/results",
    continuous_runs=5,
    continuous_max_iter=80,
    scalability_runs=3,
    scalability_dims=(10, 30, 50),
):
    output_dir = Path(output_dir)
    table_dir = Path(table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    continuous_rows = run_continuous_benchmark(
        runs=continuous_runs,
        max_iter=continuous_max_iter,
        pop_size=30,
    )
    scalability_rows = run_scalability(
        runs=scalability_runs,
        dims=scalability_dims,
        max_iter=continuous_max_iter,
        pop_size=30,
    )
    exploration_rows = run_exploration_exploitation(max_iter=continuous_max_iter, pop_size=30)

    # Discrete comparison uses classical shortest-path benchmark + TSP already available in main step 4.
    classical_rows = run_classical_search_comparison(log_func=lambda *_args, **_kwargs: None)
    _ = GridPathfindingProblem(width=25, height=25, obstacle_ratio=0.22, seed=42)

    save_convergence_plot(output_dir)
    _save_robustness_boxplot(continuous_rows, output_dir)

    _write_table(continuous_rows, table_dir / "continuous_metrics.csv")
    _write_table(scalability_rows, table_dir / "scalability_metrics.csv")
    _write_table(exploration_rows, table_dir / "exploration_metrics.csv")
    _write_table(classical_rows, table_dir / "discrete_shortest_path_metrics.csv")

    return {
        "continuous": continuous_rows,
        "scalability": scalability_rows,
        "exploration": exploration_rows,
        "discrete_shortest_path": classical_rows,
        "saved_figures": [
            str(output_dir / "requirements_convergence.png"),
            str(output_dir / "requirements_robustness.png"),
        ],
    }


if __name__ == "__main__":
    summary = run_all_requirement_benchmarks()
    print("Saved figures:")
    for p in summary["saved_figures"]:
        print(" -", p)
    print("Saved tables in experiments/results")
