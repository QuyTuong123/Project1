import numpy as np
from core.base_optimizer import BaseOptimizer
class ACO:
    def __init__(self, n_cities=20, n_ants=10, max_iter=50, rho=0.5, alpha=1, beta=2, Q=100):
        self.n_cities = n_cities
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.Q = Q

        self.coords = np.random.rand(n_cities, 2) * 100
        self.distance = self.compute_distance_matrix(self.coords)
        self.pheromone = np.ones((n_cities, n_cities))

    def compute_distance_matrix(self, coords):
        n = len(coords)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.linalg.norm(coords[i] - coords[j])
        return dist

    def tour_length(self, tour):
        total = 0
        for i in range(len(tour) - 1):
            total += self.distance[tour[i]][tour[i+1]]
        total += self.distance[tour[-1]][tour[0]]
        return total

    def select_next_city(self, current, unvisited):
        probs = []
        for city in unvisited:
            tau = self.pheromone[current][city] ** self.alpha
            eta = (1 / self.distance[current][city]) ** self.beta
            probs.append(tau * eta)

        probs = np.array(probs)
        probs = probs / probs.sum()
        return np.random.choice(unvisited, p=probs)

    def run(self):
        best_length = float("inf")
        best_tour = None

        for _ in range(self.max_iter):

            all_tours = []
            all_lengths = []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n_cities)
                tour = [start]
                unvisited = list(set(range(self.n_cities)) - {start})

                while unvisited:
                    next_city = self.select_next_city(tour[-1], unvisited)
                    tour.append(next_city)
                    unvisited.remove(next_city)

                all_tours.append(tour)
                length = self.tour_length(tour)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour

            # Update pheromone
            self.pheromone *= (1 - self.rho)

            for tour, L in zip(all_tours, all_lengths):
                for i in range(len(tour)-1):
                    self.pheromone[tour[i]][tour[i+1]] += self.Q / L

        return best_tour, best_length