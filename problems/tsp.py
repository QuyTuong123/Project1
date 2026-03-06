import numpy as np
class TSPProblem:
    def __init__(self, dim=2):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(x**2)
    
    def generate_cities(n_cities=20):
        return np.random.rand(n_cities, 2) * 100

    def compute_distance_matrix(cities):
        n = len(cities)
        dist = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist[i][j] = np.linalg.norm(cities[i] - cities[j])
        return dist

    def tour_length(tour, distance_matrix):
        total = 0
        for i in range(len(tour)):
            total += distance_matrix[tour[i]][tour[(i+1) % len(tour)]]
        return total