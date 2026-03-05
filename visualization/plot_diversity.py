import matplotlib.pyplot as plt

def plot_diversity(diversity_history):
    plt.figure()
    plt.plot(diversity_history)
    plt.title("Population Diversity")
    plt.xlabel("Iteration")
    plt.ylabel("Diversity")
    plt.show()