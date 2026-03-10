import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_surface(obj_func, lb=-5, ub=5, resolution=100, block=True):
    x = np.linspace(lb, ub, resolution)
    y = np.linspace(lb, ub, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = obj_func([X[i, j], Y[i, j]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Fitness")

    plt.title("Sphere Function Surface")
    plt.show(block=block)

def plot_particles_on_surface(obj_func, trajectory, lb=-5, ub=5, block=True):

    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)

    Z = X**2 + Y**2  # vì Sphere

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.4)

    for positions in trajectory:
        xs = positions[:, 0]
        ys = positions[:, 1]
        zs = xs**2 + ys**2
        ax.scatter(xs, ys, zs, color='r', s=10)

    plt.title("Particle Movement")
    plt.show(block=block)