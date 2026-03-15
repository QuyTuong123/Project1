import numpy as np


def bfs_search(problem):
    """Breadth-First Search on a graph/grid problem.

    Returns a dict with keys: path, cost, expanded, found.
    """
    start = problem.start
    goal = problem.goal

    frontier = [start]
    head = 0
    visited = {start}
    came_from = {}
    expanded = 0

    while head < len(frontier):
        current = frontier[head]
        head += 1
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
            frontier.append(nxt)

    return {"path": [], "cost": np.inf, "expanded": expanded, "found": False}
