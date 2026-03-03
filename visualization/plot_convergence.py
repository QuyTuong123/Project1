import matplotlib.pyplot as plt

def plot_convergence(results):
    for name, history in results.items():
        plt.plot(history, label=name)

    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.title("Convergence on Sphere Function")
    plt.show()

def plot_convergence(history, title="Convergence Plot"):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.grid(True)
    plt.show()