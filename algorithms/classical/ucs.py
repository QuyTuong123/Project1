import numpy as np


def ucs_search(problem):
    """Uniform-Cost Search on a graph/grid problem.

    Returns a dict with keys: path, cost, expanded, found.
    """
    start = problem.start
    goal = problem.goal

    open_list = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    expanded = 0

    while open_list:
        min_idx = int(np.argmin([item[0] for item in open_list]))
        current_cost, current = open_list.pop(min_idx)
        if current in visited:
            continue
        visited.add(current)
        expanded += 1

        if current == goal:
            path = problem.reconstruct_path(came_from, current)
            return {
                "path": path,
                "cost": current_cost,
                "expanded": expanded,
                "found": True,
            }

        for nxt in problem.neighbors(current):
            new_cost = current_cost + problem.step_cost(current, nxt)
            if new_cost < g_score.get(nxt, np.inf):
                g_score[nxt] = new_cost
                came_from[nxt] = current
                open_list.append((new_cost, nxt))

    return {"path": [], "cost": np.inf, "expanded": expanded, "found": False}
