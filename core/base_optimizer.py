class BaseOptimizer:
    def __init__(self, obj_func, bounds, pop_size, max_iter):
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.history = []

    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def run(self):
        self.initialize()
        for _ in range(self.max_iter):
            self.update()
            self.history.append(self.best_score)
        return self.best_solution