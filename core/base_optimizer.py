class BaseOptimizer:
    def __init__(self, obj_func, bounds, pop_size=30, max_iter=100):
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.history = []