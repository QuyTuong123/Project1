class Sphere:
    def __init__(self, dim=30):
        self.dim = dim
        self.lb = -5
        self.ub = 5
        self.global_optimum = 0

    def evaluate(self, x):
        return sum(x_i**2 for x_i in x)