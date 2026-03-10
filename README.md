# Metaheuristic Optimization Algorithms

This project implements several metaheuristic optimization algorithms for solving continuous optimization problems and the Traveling Salesman Problem (TSP).

The goal of the project is to compare the performance of different bio-inspired and evolutionary algorithms in terms of convergence, accuracy, and computational time.

---

## Implemented Algorithms

### Swarm Intelligence
- PSO — Particle Swarm Optimization
- ABC — Artificial Bee Colony
- FA — Firefly Algorithm
- CS — Cuckoo Search

### Evolutionary Algorithms
- GA — Genetic Algorithm
- DE — Differential Evolution

### Humanity Algorithms
- TLBO — Teaching Learning-based Optimization

### Ant Colony Optimization
- ACO — Ant Colony Optimization (for TSP)

---

## Project Structure

```
main.py                 # Main script to run all algorithms
algorithms/             # Implementation of optimization algorithms
├── biology/            # Bio-inspired algorithms (PSO, ABC, FA, CS, ACO)
├── evolution/          # Evolutionary algorithms (GA, DE)
├── human/              # Human-inspired algorithms (TLBO)
└── physics/            # Physics-based algorithms (SA)
core/                   # Core classes and utilities
problems/               # Optimization problems (Sphere, TSP)
experiments/            # Statistical analysis and testing
visualization/          # Plotting and visualization tools
```

---

## Requirements

- Python 3.8 or higher
- Required packages: numpy, matplotlib

---

## Installation

1. **Clone or download the project** to your local machine.

2. **Navigate to the project directory**:
   ```
   cd path/to/Metaheuristic-Optimization-Algorithms
   ```

3. **(Recommended) Create and activate a virtual environment**:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows (cmd):
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```
   - macOS / Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

4. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

   Alternatively, you can install the packages manually:
   ```
   pip install numpy matplotlib
   ```

---

## Running the Program

1. **Run the main script**:
   ```
   python main.py
   ```

2. The program will execute the following steps:
   - Compare continuous optimization algorithms on a 2D Sphere problem
   - Generate visualizations (convergence plots, diversity plots, 3D surface plots)
   - Perform statistical comparison (mean, standard deviation, runtime)
   - Solve TSP using Ant Colony Optimization

3. **Output**:
   - Console output showing algorithm comparisons and results
   - Generated plots saved in the project directory (if visualization is enabled)

---

## Usage Examples

### Running Individual Algorithms

You can modify `main.py` to run specific algorithms or change parameters:

```python
from algorithms.biology.pso import PSO
from problems.sphere import Sphere

problem = Sphere(dim=2)
optimizer = PSO(
    obj_func=problem.evaluate,
    bounds=(problem.lb, problem.ub),
    dim=2,
    pop_size=30,
    max_iter=100
)
best_position, best_score = optimizer.run()
print(f"Best position: {best_position}, Best score: {best_score}")
```

### Running Experiments

For statistical analysis:

```python
from experiments.convergence_analysis import run_30_times
from algorithms.biology.abc import ABC

mean, std, runtime = run_30_times(ABC)
print(f"Mean: {mean}, Std: {std}, Time: {runtime}")
```

### Visualization

```python
from visualization.plot_convergence import plot_convergence
from algorithms.biology.abc import ABC

optimizer = ABC(...)
optimizer.run()
plot_convergence(optimizer.history)
```

---

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Python version**: Make sure you're using Python 3.8+
- **Matplotlib issues**: If plots don't display, ensure you have a GUI backend installed (e.g., Tkinter)

---

## Contributing

Feel free to contribute by adding new algorithms, problems, or improving existing implementations.

---

## License

This project is open-source. Please refer to the license file for details.
