import time
import numpy as np
from algorithms.pso import PSO
from problems.continuous_functions import sphere

dim = 30
bounds = (-5, 5)

results = []
times = []

for _ in range(30):
    start = time.time()
    opt = PSO(sphere, bounds, dim=dim)
    best = opt.run()
    end = time.time()

    results.append(best)
    times.append(end - start)

mean = np.mean(results)
std = np.std(results)
avg_time = np.mean(times)

print("Mean:", mean)
print("Std:", std)
print("Time:", avg_time)