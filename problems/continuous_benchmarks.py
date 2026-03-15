import numpy as np


class ContinuousBenchmark:
    def __init__(self, dim=30, lb=-5.0, ub=5.0, name="benchmark"):
        self.dim = dim
        self.lb = np.full(dim, lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float)
        self.name = name

    def evaluate(self, x):
        raise NotImplementedError


class SphereBenchmark(ContinuousBenchmark):
    def __init__(self, dim=30):
        super().__init__(dim=dim, lb=-5.0, ub=5.0, name="Sphere")

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x ** 2))


class RastriginBenchmark(ContinuousBenchmark):
    def __init__(self, dim=30):
        super().__init__(dim=dim, lb=-5.12, ub=5.12, name="Rastrigin")

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        n = x.size
        return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


class RosenbrockBenchmark(ContinuousBenchmark):
    def __init__(self, dim=30):
        super().__init__(dim=dim, lb=-2.0, ub=2.0, name="Rosenbrock")

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


class GriewankBenchmark(ContinuousBenchmark):
    def __init__(self, dim=30):
        super().__init__(dim=dim, lb=-600.0, ub=600.0, name="Griewank")

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        i = np.sqrt(np.arange(1, x.size + 1, dtype=float))
        return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / i)) + 1.0)


class AckleyBenchmark(ContinuousBenchmark):
    def __init__(self, dim=30):
        super().__init__(dim=dim, lb=-32.768, ub=32.768, name="Ackley")

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        n = x.size
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(2.0 * np.pi * x))
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        return float(term1 + term2 + 20.0 + np.e)
