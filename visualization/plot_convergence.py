import matplotlib.pyplot as plt

def plot_multiple_convergence(results, block=True):
    for name, history in results.items():
        plt.plot(history, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.title("Convergence Comparison")
    plt.grid(True)
    plt.show(block=block)

def plot_convergence(history, title="Convergence Plot", block=True):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.grid(True)
    plt.show(block=block)

def plot_diversity(diversity_history, title="Exploration Analysis", block=True):
    plt.figure()
    plt.plot(diversity_history)
    plt.xlabel("Iteration")
    plt.ylabel("Diversity (Std)")
    plt.title(title)
    plt.grid(True)
    plt.show(block=block)