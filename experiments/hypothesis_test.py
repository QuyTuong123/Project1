import numpy as np

from algorithms.physics.hill_climbing import HillClimbing
from algorithms.physics.sa import SA
from problems.sphere import Sphere


def _collect_scores(optimizer_factory, runs=30):
    scores = []
    for _ in range(runs):
        opt = optimizer_factory()
        _, score = opt.run()
        scores.append(float(score))
    return np.asarray(scores, dtype=float)


def permutation_test_mean_diff(a, b, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    observed = abs(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        pa = pooled[:n_a]
        pb = pooled[n_a:]
        stat = abs(np.mean(pa) - np.mean(pb))
        if stat >= observed:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return observed, float(p_value)


def run_hc_vs_sa_hypothesis_test(runs=30, dim=30, max_iter=200):
    problem = Sphere(dim=dim)

    hc_scores = _collect_scores(
        lambda: HillClimbing(
            obj_func=problem.evaluate,
            bounds=(problem.lb, problem.ub),
            dim=problem.dim,
            pop_size=30,
            max_iter=max_iter,
            step_size=0.1,
            n_neighbors=20,
        ),
        runs=runs,
    )

    sa_scores = _collect_scores(
        lambda: SA(
            obj_func=problem.evaluate,
            bounds=(problem.lb, problem.ub),
            dim=problem.dim,
            pop_size=30,
            max_iter=max_iter,
            initial_temperature=100.0,
            cooling_rate=0.95,
            step_sigma=0.1,
        ),
        runs=runs,
    )

    effect, p_value = permutation_test_mean_diff(hc_scores, sa_scores)
    return {
        "hc_mean": float(np.mean(hc_scores)),
        "sa_mean": float(np.mean(sa_scores)),
        "mean_abs_diff": float(effect),
        "p_value": p_value,
        "significant_at_0_05": bool(p_value < 0.05),
    }


if __name__ == "__main__":
    out = run_hc_vs_sa_hypothesis_test()
    print("HC mean:", f"{out['hc_mean']:.6e}")
    print("SA mean:", f"{out['sa_mean']:.6e}")
    print("|mean diff|:", f"{out['mean_abs_diff']:.6e}")
    print("p-value:", f"{out['p_value']:.6f}")
    print("Significant at 0.05:", out["significant_at_0_05"])
