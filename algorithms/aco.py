import numpy as np
import matplotlib.pyplot as plt
import time

class ACO:
    def __init__(self, n_cities, n_ants, rho):
        self.n_cities = n_cities
        self.n_ants = n_ants
        self.rho = rho

pheromone = np.ones((n_cities, n_cities))

coords = np.random.rand(n_cities, 2) * 100
def compute_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist



def select_next_city(current, unvisited, pheromone, distance, alpha, beta):
    probs = []
    for city in unvisited:
        tau = pheromone[current][city] ** alpha
        eta = (1 / distance[current][city]) ** beta
        probs.append(tau * eta)

    probs = np.array(probs)
    probs = probs / probs.sum()

    return np.random.choice(unvisited, p=probs)

def tour_length(tour, distance):
    total = 0
    for i in range(len(tour)-1):
        total += distance[tour[i]][tour[i+1]]
    total += distance[tour[-1]][tour[0]]
    return total

for iteration in range(max_iter):

    all_tours = []
    all_lengths = []

    for ant in range(n_ants):

        start = np.random.randint(n_cities)
        tour = [start]

        unvisited = list(set(range(n_cities)) - {start})

        while unvisited:
            next_city = select_next_city(
                tour[-1], unvisited,
                pheromone, distance,
                alpha, beta
            )
            tour.append(next_city)
            unvisited.remove(next_city)

        all_tours.append(tour)
        all_lengths.append(tour_length(tour, distance))

    # Update pheromone
    pheromone *= (1 - rho)

    for tour, L in zip(all_tours, all_lengths):
        for i in range(len(tour)-1):
            pheromone[tour[i]][tour[i+1]] += Q / L
