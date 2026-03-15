import numpy as np


def greedy_best_first_search(problem):
    """Greedy Best-First Search on a graph/grid problem.

    Returns a dict with keys: path, cost, expanded, found.
    """
    start = problem.start
    goal = problem.goal

    open_list = [(problem.heuristic(start), start)]
    visited = {start}
    came_from = {}
    expanded = 0

    while open_list:
        min_idx = int(np.argmin([item[0] for item in open_list]))
        _, current = open_list.pop(min_idx)
        expanded += 1
        if current == goal:
            path = problem.reconstruct_path(came_from, current)
            return {
                "path": path,
                "cost": problem.path_cost(path),
                "expanded": expanded,
                "found": True,
            }

        for nxt in problem.neighbors(current):
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            open_list.append((problem.heuristic(nxt), nxt))

    return {"path": [], "cost": np.inf, "expanded": expanded, "found": False}
