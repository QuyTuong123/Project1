import time
import threading
import numpy as np

from problems.sphere import Sphere
from problems.tsp import TSPProblem

# Biology algorithms
from algorithms.biology.abc import ABC
from algorithms.biology.pso import PSO
from algorithms.biology.aco import ACO
from algorithms.biology.fa import FA
from algorithms.biology.cs import CS

# Evolution algorithms
from algorithms.evolution.ga import GA
from algorithms.evolution.de import DE

# Physics algorithms
from algorithms.physics.sa import SA

# Human algorithms
from algorithms.human.tlbo import TLBO

# Visualization
from experiments.convergence_analysis import run_30_times
from visualization.plot_convergence import plot_convergence, plot_diversity
from visualization.plot_3d_surface import plot_surface, plot_particles_on_surface

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    tk = None


def format_position(pos):
    pos_str = "["
    for i, x in enumerate(pos):
        if abs(x) < 1e-4:
            val = f"{x:.1e}"
        else:
            val = f"{x:.5f}"

        if i < len(pos) - 1:
            pos_str += val + " , "
        else:
            pos_str += val
    pos_str += "]"
    return pos_str


class Runner:
    def __init__(self, log_func=print, tk_root=None):
        self.log = log_func
        self.tk_root = tk_root
        self.problem2d = Sphere(dim=2)

    def step_1(self):
        self.log("\n===== STEP 1: COMPARE CONTINUOUS ALGORITHMS =====")
        algorithms = {
            "FA": FA,
            "ABC": ABC,
            "CS": CS,
            "PSO": PSO,
            "GA": GA,
            "DE": DE,
            "SA": SA,
            "TLBO": TLBO,
        }

        results = {}
        for name, Algo in algorithms.items():
            optimizer = Algo(
                obj_func=self.problem2d.evaluate,
                bounds=(self.problem2d.lb, self.problem2d.ub),
                dim=2,
                pop_size=30,
                max_iter=100,
            )
            best_position, best_score = optimizer.run()
            results[name] = (best_score, best_position)

        self.log("\n===== TABLE 1: BEST FITNESS (DIM=2) =====")
        self.log("{:<10} {:<18} {}".format("Algorithm", "Best Fitness", "Best Position"))
        self.log("-" * 60)

        for name, (fit, pos) in results.items():
            fit_str = f"{fit:.2e}"
            pos_str = format_position(pos)
            self.log("{:<10} {:<18} {}".format(name, fit_str, pos_str))

        self.log("Done STEP 1")

    def step_2(self):
        self.log("\n===== STEP 2: VISUALIZATION (4 HISTOGRAMS) =====")
        optimizer = ABC(
            obj_func=self.problem2d.evaluate,
            bounds=(self.problem2d.lb, self.problem2d.ub),
            dim=2,
            pop_size=30,
            max_iter=100,
        )
        optimizer.run()

        if self.tk_root is None:
            # Console mode: render one by one (blocking) as before
            plot_convergence(optimizer.history)
            plot_diversity(optimizer.diversity_history)
            plot_surface(self.problem2d.evaluate, self.problem2d.lb, self.problem2d.ub)
            plot_particles_on_surface(self.problem2d.evaluate, optimizer.trajectory)
        else:
            # Ensure all Tkinter / Matplotlib windows are created on the main GUI thread.
            self.tk_root.after(0, lambda: self._show_visualization_menu(optimizer))

        self.log("Done STEP 2")

    def _show_visualization_menu(self, optimizer):
        win = tk.Toplevel(self.tk_root)
        win.title("Visualization - Choose plots")
        win.geometry("360x260")

        ttk.Label(win, text="Click a plot to display it:", font=(None, 12)).pack(anchor=tk.W, padx=10, pady=(10, 4))

        button_frame = ttk.Frame(win)
        button_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        def add_plot_button(label, func):
            b = ttk.Button(button_frame, text=label, command=lambda: self._safe_plot(func))
            b.pack(fill=tk.X, pady=4)

        add_plot_button("Convergence", lambda: plot_convergence(optimizer.history, block=False))
        add_plot_button("Diversity", lambda: plot_diversity(optimizer.diversity_history, block=False))
        add_plot_button("Surface", lambda: plot_surface(self.problem2d.evaluate, self.problem2d.lb, self.problem2d.ub, block=False))
        add_plot_button("Particle Movement", lambda: plot_particles_on_surface(self.problem2d.evaluate, optimizer.trajectory, block=False))

        ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8, padx=10)

        ttk.Button(win, text="Close", command=win.destroy).pack(fill=tk.X, padx=10, pady=(0, 10))

    def _safe_plot(self, func):
        try:
            func()
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def step_3(self):
        self.log("\n===== STEP 3: STATISTIC COMPARISON =====")
        algos_8 = {
            "FA": FA,
            "ABC": ABC,
            "CS": CS,
            "PSO": PSO,
            "GA": GA,
            "DE": DE,
            "SA": SA,
            "TLBO": TLBO,
        }

        self.log("\n===== TABLE 2: MEAN / STD / TIME =====")
        self.log("{:<10} {:<15} {:<15} {:<10}".format("Algorithm", "Mean", "Std", "Time(s)"))
        self.log("-" * 55)

        for name, Algo in algos_8.items():
            start = time.time()
            mean, std, runtime = run_30_times(Algo)
            end = time.time()
            runtime = end - start
            self.log("{:<10} {:<15.6e} {:<15.6e} {:<10.4f}".format(name, mean, std, runtime))

        self.log("Done STEP 3")

    def step_4(self):
        self.log("\n===== STEP 4: TSP WITH ACO (DISCRETE OPTIMIZATION) =====")
        aco = ACO(n_cities=20, n_ants=10, max_iter=50)
        best_tour, best_distance = aco.run()
        self.log("Best Tour: %s" % best_tour)
        self.log("Best Distance: %s" % best_distance)
        self.log("Done STEP 4")


class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Demo - Main Menu")
        self.geometry("700x520")

        self._create_widgets()

    def _create_widgets(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Select a step to run:", font=(None, 14)).pack(anchor=tk.W, pady=(0, 8))

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.buttons = []
        runner = Runner(log_func=self._log, tk_root=self)

        for text, step in [
            ("Step 1: Compare Algorithms", runner.step_1),
            ("Step 2: Visualization", runner.step_2),
            ("Step 3: Statistics", runner.step_3),
            ("Step 4: TSP (ACO)", runner.step_4),
        ]:
            b = ttk.Button(button_frame, text=text)
            b.config(command=lambda s=step, btn=b: self._run_in_thread(s, btn))
            b.pack(side=tk.LEFT, padx=4, pady=4, expand=True, fill=tk.X)
            self.buttons.append(b)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.output = tk.Text(frame, wrap=tk.WORD, height=18)
        self.output.pack(fill=tk.BOTH, expand=True)

        self._log("Ready. Click a step button to run it.")

    def _log(self, message: str):
        self.output.insert(tk.END, message + "\n")
        self.output.see(tk.END)

    def _run_in_thread(self, func, button):
        # Disable only the button that started the task so other steps can still be run
        button.config(state=tk.DISABLED)

        def task():
            try:
                func()
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                button.config(state=tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def _set_buttons_state(self, state):
        for b in self.buttons:
            b.config(state=state)


def main():
    if tk is None:
        print("Tkinter is not available. Running in console mode.")
        Runner().step_1()
        Runner().step_2()
        Runner().step_3()
        Runner().step_4()
        return

    app = MainMenu()
    app.mainloop()


if __name__ == "__main__":
    main()