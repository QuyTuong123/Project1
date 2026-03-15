import numpy as np


class GridPathfindingProblem:
    """Simple 4-neighborhood grid pathfinding problem for classical search comparison."""

    def __init__(self, width=20, height=20, obstacle_ratio=0.2, seed=42):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)

        rng = np.random.default_rng(seed)
        self.obstacles = set()
        for r in range(height):
            for c in range(width):
                if (r, c) in (self.start, self.goal):
                    continue
                if float(rng.random()) < obstacle_ratio:
                    self.obstacles.add((r, c))

        # Ensure there is always at least one simple corridor from start to goal.
        for r in range(height):
            self.obstacles.discard((r, 0))
        for c in range(width):
            self.obstacles.discard((height - 1, c))

    def in_bounds(self, node):
        r, c = node
        return 0 <= r < self.height and 0 <= c < self.width

    def passable(self, node):
        return node not in self.obstacles

    def neighbors(self, node):
        r, c = node
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        result = []
        for nxt in candidates:
            if self.in_bounds(nxt) and self.passable(nxt):
                result.append(nxt)
        return result

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def step_cost(self, _a, _b):
        return 1.0

    @staticmethod
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def path_cost(self, path):
        if not path:
            return np.inf
        return float(len(path) - 1)
