from pathlib import Path
import time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from algorithms.physics.hill_climbing import HillClimbing
from algorithms.physics.sa import SA
from problems.sphere import Sphere


def _run_many(factory, runs=20):
    scores = []
    t0 = time.perf_counter()
    for _ in range(runs):
        opt = factory()
        _, best = opt.run()
        scores.append(best)
    dt = time.perf_counter() - t0
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), float(dt / runs)


def analyze_hc_step_size(runs=20, dim=30):
    problem = Sphere(dim=dim)
    step_sizes = [0.02, 0.05, 0.1, 0.2, 0.4]
    rows = []

    for step in step_sizes:
        mean, std, avg_time = _run_many(
            lambda: HillClimbing(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim=problem.dim,
                pop_size=30,
                max_iter=200,
                step_size=step,
                n_neighbors=20,
            ),
            runs=runs,
        )
        rows.append({"param": step, "mean": mean, "std": std, "avg_time": avg_time})

    return rows


def analyze_sa_cooling_rate(runs=20, dim=30):
    problem = Sphere(dim=dim)
    cooling_rates = [0.85, 0.9, 0.93, 0.95, 0.98]
    rows = []

    for rate in cooling_rates:
        mean, std, avg_time = _run_many(
            lambda: SA(
                obj_func=problem.evaluate,
                bounds=(problem.lb, problem.ub),
                dim=problem.dim,
                pop_size=30,
                max_iter=200,
                initial_temperature=100.0,
                cooling_rate=rate,
                step_sigma=0.1,
            ),
            runs=runs,
        )
        rows.append({"param": rate, "mean": mean, "std": std, "avg_time": avg_time})

    return rows


def _plot_errorbar(rows, title, xlabel, output_path):
    x = [r["param"] for r in rows]
    y = [r["mean"] for r in rows]
    yerr = [r["std"] for r in rows]

    fig = Figure(figsize=(7, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Best Score (+/- std)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)


def _write_csv(rows, file_path):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["param,mean,std,avg_time"]
    for r in rows:
        lines.append(f"{r['param']},{r['mean']:.6e},{r['std']:.6e},{r['avg_time']:.6e}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parameter_sensitivity(output_dir="visualizes", table_dir="experiments/results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hc_rows = analyze_hc_step_size(runs=20, dim=30)
    sa_rows = analyze_sa_cooling_rate(runs=20, dim=30)

    _plot_errorbar(
        hc_rows,
        title="HC Parameter Sensitivity (step_size)",
        xlabel="step_size",
        output_path=output_dir / "sensitivity_hc_step_size.png",
    )
    _plot_errorbar(
        sa_rows,
        title="SA Parameter Sensitivity (cooling_rate)",
        xlabel="cooling_rate",
        output_path=output_dir / "sensitivity_sa_cooling_rate.png",
    )

    _write_csv(hc_rows, Path(table_dir) / "sensitivity_hc_step_size.csv")
    _write_csv(sa_rows, Path(table_dir) / "sensitivity_sa_cooling_rate.csv")

    return {
        "hc": hc_rows,
        "sa": sa_rows,
        "figures": [
            str(output_dir / "sensitivity_hc_step_size.png"),
            str(output_dir / "sensitivity_sa_cooling_rate.png"),
        ],
    }


if __name__ == "__main__":
    out = run_parameter_sensitivity()
    print("Saved sensitivity figures:")
    for fp in out["figures"]:
        print(" -", fp)
