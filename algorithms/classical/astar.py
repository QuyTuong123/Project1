import numpy as np


def astar_search(problem):
    """A* Search on a graph/grid problem.

    Returns a dict with keys: path, cost, expanded, found.
    """
    start = problem.start
    goal = problem.goal

    open_list = [(problem.heuristic(start), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    expanded = 0

    while open_list:
        min_idx = int(np.argmin([item[0] for item in open_list]))
        _, current_g, current = open_list.pop(min_idx)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            path = problem.reconstruct_path(came_from, current)
            return {
                "path": path,
                "cost": current_g,
                "expanded": expanded,
                "found": True,
            }

        for nxt in problem.neighbors(current):
            tentative_g = current_g + problem.step_cost(current, nxt)
            if tentative_g < g_score.get(nxt, np.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                f_score = tentative_g + problem.heuristic(nxt)
                open_list.append((f_score, tentative_g, nxt))

    return {"path": [], "cost": np.inf, "expanded": expanded, "found": False}
