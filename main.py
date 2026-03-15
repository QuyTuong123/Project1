import time
import threading
import numpy as np
from pathlib import Path

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
from algorithms.physics.hill_climbing import HillClimbing

# Human algorithms
from algorithms.human.tlbo import TLBO

# Visualization
from experiments.convergence_analysis import run_30_times
from experiments.spec_comparison import run_spec_comparison
from experiments.benchmark_requirements import run_all_requirement_benchmarks
from experiments.parameter_sensitivity import run_parameter_sensitivity
from experiments.hypothesis_test import run_hc_vs_sa_hypothesis_test
from visualization.plot_convergence import plot_convergence, plot_diversity, plot_multiple_convergence
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
        self.log("\n===== STEP 1: COMPARE ALGORITHMS =====")
        algorithms = {
            "FA": FA,
            "ABC": ABC,
            "CS": CS,
            "PSO": PSO,
            "GA": GA,
            "DE": DE,
            "HC": HillClimbing,
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

        # Merge project-spec comparison (classical + local-search) into Step 1.
        run_spec_comparison(log_func=self.log)

        self.log("Done STEP 1")

    def step_2(self):
        self.log("\n===== STEP 2: CONCRETE VISUALIZATIONS (REQUIREMENT-ALIGNED) =====")
        self.log("Preparing data for concrete plots: ABC (2D Sphere) + SA/HC (30D Sphere).")

        abc = ABC(
            obj_func=self.problem2d.evaluate,
            bounds=(self.problem2d.lb, self.problem2d.ub),
            dim=2,
            pop_size=30,
            max_iter=100,
        )
        abc.run()

        problem30 = Sphere(dim=30)
        sa = SA(
            obj_func=problem30.evaluate,
            bounds=(problem30.lb, problem30.ub),
            dim=30,
            max_iter=120,
            initial_temperature=100.0,
            cooling_rate=0.95,
            step_sigma=0.1,
        )
        sa.run()

        hc = HillClimbing(
            obj_func=problem30.evaluate,
            bounds=(problem30.lb, problem30.ub),
            dim=30,
            max_iter=120,
            step_size=0.1,
            n_neighbors=20,
        )
        hc.run()

        if self.tk_root is None:
            # Console mode: render concrete requirement plots one by one.
            plot_convergence(abc.history, title="ABC Convergence on Sphere (2D)")
            plot_diversity(abc.diversity_history, title="ABC Diversity on Sphere (2D)")
            plot_surface(
                self.problem2d.evaluate,
                self.problem2d.lb,
                self.problem2d.ub,
                title="Sphere Objective Landscape (2D -> 3D)",
            )
            plot_particles_on_surface(
                self.problem2d.evaluate,
                abc.trajectory,
                title="ABC Particle Trajectory on Sphere Surface (2D)",
            )
            plot_multiple_convergence(
                {
                    "SA (30D Sphere)": sa.history,
                    "Hill Climbing (30D Sphere)": hc.history,
                }
            )
            plot_convergence(sa.history, title="SA Convergence on Sphere (30D)")
            plot_convergence(hc.history, title="Hill Climbing Convergence on Sphere (30D)")
        else:
            # Ensure all Tkinter / Matplotlib windows are created on the main GUI thread.
            try:
                self.tk_root.after(
                    0,
                    lambda: self._show_visualization_menu(
                        abc=abc,
                        sa_history=sa.history,
                        hc_history=hc.history,
                    ),
                )
            except (RuntimeError, tk.TclError):
                # UI may already be closing; skip opening visualization menu.
                pass

        self.log("Done STEP 2")

    def _show_visualization_menu(self, abc, sa_history, hc_history):
        win = tk.Toplevel(self.tk_root)
        win.title("Step 2 - Concrete Required Visualizations")
        win.geometry("540x430")

        ttk.Label(
            win,
            text="Choose specific visualizations (not generic placeholders):",
            font=(None, 12),
        ).pack(anchor=tk.W, padx=10, pady=(10, 4))

        button_frame = ttk.Frame(win)
        button_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        def add_plot_button(label, func):
            b = ttk.Button(button_frame, text=label, command=lambda: self._safe_plot(func))
            b.pack(fill=tk.X, pady=4)

        add_plot_button(
            "ABC Convergence on Sphere (2D)",
            lambda: plot_convergence(abc.history, title="ABC Convergence on Sphere (2D)", block=False),
        )
        add_plot_button(
            "ABC Diversity on Sphere (2D)",
            lambda: plot_diversity(abc.diversity_history, title="ABC Diversity on Sphere (2D)", block=False),
        )
        add_plot_button(
            "Sphere Objective Landscape (2D -> 3D)",
            lambda: plot_surface(
                self.problem2d.evaluate,
                self.problem2d.lb,
                self.problem2d.ub,
                title="Sphere Objective Landscape (2D -> 3D)",
                block=False,
            ),
        )
        add_plot_button(
            "ABC Particle Trajectory on Sphere Surface (2D)",
            lambda: plot_particles_on_surface(
                self.problem2d.evaluate,
                abc.trajectory,
                title="ABC Particle Trajectory on Sphere Surface (2D)",
                block=False,
            ),
        )
        add_plot_button(
            "SA vs Hill Climbing Convergence (30D)",
            lambda: plot_multiple_convergence(
                {
                    "SA (30D Sphere)": sa_history,
                    "Hill Climbing (30D Sphere)": hc_history,
                },
                block=False,
            ),
        )
        add_plot_button(
            "SA Convergence on Sphere (30D)",
            lambda: plot_convergence(sa_history, title="SA Convergence on Sphere (30D)", block=False),
        )
        add_plot_button(
            "Hill Climbing Convergence on Sphere (30D)",
            lambda: plot_convergence(hc_history, title="Hill Climbing Convergence on Sphere (30D)", block=False),
        )

        def show_all_required():
            plot_convergence(abc.history, title="ABC Convergence on Sphere (2D)", block=False)
            plot_diversity(abc.diversity_history, title="ABC Diversity on Sphere (2D)", block=False)
            plot_surface(
                self.problem2d.evaluate,
                self.problem2d.lb,
                self.problem2d.ub,
                title="Sphere Objective Landscape (2D -> 3D)",
                block=False,
            )
            plot_particles_on_surface(
                self.problem2d.evaluate,
                abc.trajectory,
                title="ABC Particle Trajectory on Sphere Surface (2D)",
                block=False,
            )
            plot_multiple_convergence(
                {
                    "SA (30D Sphere)": sa_history,
                    "Hill Climbing (30D Sphere)": hc_history,
                },
                block=False,
            )
            plot_convergence(sa_history, title="SA Convergence on Sphere (30D)", block=False)
            plot_convergence(hc_history, title="Hill Climbing Convergence on Sphere (30D)", block=False)

        ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8, padx=10)
        ttk.Button(
            win,
            text="Show All Required Visualizations",
            command=lambda: self._safe_plot(show_all_required),
        ).pack(fill=tk.X, padx=10, pady=(0, 6))

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
            "HC": HillClimbing,
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
        clean_tour = [int(city) for city in best_tour]
        self.log("Best Tour: %s" % clean_tour)
        self.log("Best Distance: %s" % best_distance)
        self.log("Done STEP 4")

    def core_step_1_required_compare(self):
        self.log("\n===== CORE STEP 1 (REQUIREMENT): ALGORITHM COVERAGE + BASELINE COMPARISON =====")
        self.log("Goal: Verify required algorithms and compare with classical/local baselines.")
        self.step_1()
        self.log("Done CORE STEP 1")

    def core_step_2_required_visualization(self):
        self.log("\n===== CORE STEP 2 (REQUIREMENT): VISUALIZATION =====")
        self.log("Goal: Convergence/diversity/3D surface + SA/HC convergence figures.")
        self.step_2()
        self.generate_local_search_figures()
        self.log("Done CORE STEP 2")

    def core_step_3_required_metrics(self):
        self.log("\n===== CORE STEP 3 (REQUIREMENT): METRICS =====")
        self.log("Goal: mean/std/time + convergence speed + memory + scalability + exploration/exploitation.")
        self.step_3()
        self.run_requirements_quick()
        self.log("Done CORE STEP 3")

    def core_step_4_required_advanced(self):
        self.log("\n===== CORE STEP 4 (REQUIREMENT): SENSITIVITY + HYPOTHESIS + DISCRETE =====")
        self.log("Goal: parameter sensitivity, statistical hypothesis test, and discrete TSP output.")
        self.run_parameter_sensitivity()
        self.run_hypothesis_test()
        self.step_4()
        self.log("Done CORE STEP 4")

    def step_1_to_4(self):
        self.log("\n===== RUN CORE WORKFLOW (REQUIREMENT-BASED): STEP 1 -> STEP 4 =====")
        self.core_step_1_required_compare()
        self.core_step_2_required_visualization()
        self.core_step_3_required_metrics()
        self.core_step_4_required_advanced()
        self.log("Done CORE WORKFLOW")

    def run_requirements_quick(self):
        self.log("\n===== REQUIREMENT BENCHMARK (QUICK) =====")
        run_all_requirement_benchmarks(
            continuous_runs=2,
            continuous_max_iter=30,
            scalability_runs=2,
            scalability_dims=(10, 30),
        )
        self.log("Saved CSV tables in experiments/results")
        self.log("Done REQUIREMENT BENCHMARK (QUICK)")

    def run_requirements_full(self):
        self.log("\n===== REQUIREMENT BENCHMARK (FULL) =====")
        run_all_requirement_benchmarks(
            continuous_runs=5,
            continuous_max_iter=80,
            scalability_runs=3,
            scalability_dims=(10, 30, 50),
        )
        self.log("Saved CSV tables in experiments/results")
        self.log("Done REQUIREMENT BENCHMARK (FULL)")

    def run_parameter_sensitivity(self):
        self.log("\n===== PARAMETER SENSITIVITY =====")
        run_parameter_sensitivity()
        self.log("Saved CSV tables in experiments/results")
        self.log("Done PARAMETER SENSITIVITY")

    def run_hypothesis_test(self):
        self.log("\n===== HYPOTHESIS TEST: HC VS SA =====")
        out = run_hc_vs_sa_hypothesis_test(runs=30, dim=30, max_iter=200)
        self.log(f"HC mean: {out['hc_mean']:.6e}")
        self.log(f"SA mean: {out['sa_mean']:.6e}")
        self.log(f"|mean diff|: {out['mean_abs_diff']:.6e}")
        self.log(f"p-value: {out['p_value']:.6f}")
        self.log(f"Significant at 0.05: {out['significant_at_0_05']}")
        self.log("Done HYPOTHESIS TEST")

    def _save_single_convergence(self, algo_cls, title, out_name):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        problem = Sphere(dim=30)
        optimizer = algo_cls(
            obj_func=problem.evaluate,
            bounds=(problem.lb, problem.ub),
            dim=30,
            pop_size=30,
            max_iter=200,
        )
        optimizer.run()

        output_dir = Path("visualizes")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / out_name

        fig = Figure(figsize=(7, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(optimizer.history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness")
        ax.set_title(title)
        ax.grid(True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return str(output_path)

    def generate_local_search_figures(self):
        self.log("\n===== GENERATE SA/HC CONVERGENCE FIGURES =====")
        self._save_single_convergence(SA, "SA Convergence", "local_search_sa_convergence.png")
        self._save_single_convergence(HillClimbing, "Hill Climbing Convergence", "local_search_hc_convergence.png")
        self.log("Done GENERATE SA/HC CONVERGENCE FIGURES")

    def run_all_for_report(self):
        self.log("\n===== RUN ALL FOR REPORT =====")
        self.step_1_to_4()
        self.log("Done RUN ALL FOR REPORT")

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Project Control Center")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Idle")
        self.runner = Runner(log_func=self._log, tk_root=self)
        self.buttons = []
        self._closing = False

        self._create_widgets()

    def _create_widgets(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Run all project features from one place",
            font=(None, 14, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text=(
                "Core steps, requirement benchmarks, sensitivity analysis, hypothesis test, "
                "and report figure generation are grouped by tabs below."
            ),
        ).pack(anchor=tk.W, pady=(2, 10))

        main_split = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        main_split.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_split, padding=(0, 0, 8, 0))
        right = ttk.Frame(main_split)
        main_split.add(left, weight=2)
        main_split.add(right, weight=3)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_core = ttk.Frame(notebook, padding=8)
        tab_requirements = ttk.Frame(notebook, padding=8)
        tab_all = ttk.Frame(notebook, padding=8)
        notebook.add(tab_core, text="Core Workflow")
        notebook.add(tab_requirements, text="Requirement Tools")
        notebook.add(tab_all, text="One Click")

        self._build_action_group(
            tab_core,
            "Requirement-Aligned Core Steps",
            [
                (
                    "Step 1: Required Algorithms + Baseline Compare",
                    self.runner.core_step_1_required_compare,
                    "Nature-inspired algorithms + BFS/DFS/UCS/Greedy/A*/HC/SA comparison",
                ),
                (
                    "Step 2: Required Visualizations",
                    self.runner.core_step_2_required_visualization,
                    "Convergence, diversity, 3D landscape, and SA/HC convergence figure outputs",
                ),
                (
                    "Step 3: Required Metrics",
                    self.runner.core_step_3_required_metrics,
                    "Convergence speed, quality, robustness, time/space, scalability, exploration/exploitation",
                ),
                (
                    "Step 4: Sensitivity + Hypothesis + Discrete",
                    self.runner.core_step_4_required_advanced,
                    "Parameter sensitivity, statistical significance, and discrete TSP result",
                ),
                (
                    "Run Core Step 1 -> 4",
                    self.runner.step_1_to_4,
                    "Run the full requirement-based core workflow in order",
                ),
            ],
        )

        self._build_action_group(
            tab_requirements,
            "Assignment Requirement Coverage",
            [
                ("Generate SA/HC Convergence Figures", self.runner.generate_local_search_figures, "Save local_search_sa/hc_convergence.png"),
                ("Requirement Benchmark (Quick)", self.runner.run_requirements_quick, "Fast check run for all metrics and outputs"),
                ("Requirement Benchmark (Full)", self.runner.run_requirements_full, "Higher-fidelity run for final report"),
                ("Parameter Sensitivity", self.runner.run_parameter_sensitivity, "HC step_size + SA cooling_rate"),
                ("Hypothesis Test (HC vs SA)", self.runner.run_hypothesis_test, "Permutation test with p-value"),
            ],
        )

        self._build_action_group(
            tab_all,
            "Full Pipeline",
            [
                (
                    "Run All For Report",
                    self.runner.run_all_for_report,
                    "Step 1, Step 3, Step 4, figures, requirement quick benchmark, sensitivity, hypothesis",
                ),
            ],
        )

        status_row = ttk.Frame(left)
        status_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(status_row, text="Status:", font=(None, 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(6, 0))

        log_top = ttk.Frame(right)
        log_top.pack(fill=tk.X)
        ttk.Label(log_top, text="Execution Log", font=(None, 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(log_top, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)

        self.output = tk.Text(right, wrap=tk.WORD, height=24)
        self.output.pack(fill=tk.BOTH, expand=True)

        self._log("Ready. Select any action from tabs and run.")

    def _build_action_group(self, parent, title, items):
        group = ttk.LabelFrame(parent, text=title, padding=8)
        group.pack(fill=tk.BOTH, expand=True)
        for text, action, desc in items:
            row = ttk.Frame(group)
            row.pack(fill=tk.X, pady=4)
            btn = ttk.Button(row, text=text)
            btn.config(command=lambda s=action, b=btn, t=text: self._run_in_thread(s, b, t))
            btn.pack(fill=tk.X)
            self.buttons.append(btn)
            ttk.Label(row, text=desc, foreground="#555555").pack(anchor=tk.W, pady=(2, 0))

    def _append_log(self, message: str):
        if self._closing:
            return
        try:
            self.output.insert(tk.END, message + "\n")
            self.output.see(tk.END)
        except tk.TclError:
            # Window may already be closing/destroyed.
            pass

    def _call_on_ui(self, callback):
        """Safely schedule callback on Tk main loop while app is alive."""
        if self._closing:
            return
        try:
            self.after(0, callback)
        except RuntimeError:
            # Main loop is no longer running.
            pass
        except tk.TclError:
            # Window already destroyed.
            pass

    def _log(self, message: str):
        # Allow safe logging from worker threads.
        if threading.current_thread() is threading.main_thread():
            self._append_log(message)
        else:
            self._call_on_ui(lambda m=message: self._append_log(m))

    def _clear_log(self):
        self.output.delete("1.0", tk.END)

    def _run_in_thread(self, func, button, label):
        if self._closing:
            return

        # Disable only the button that started the task so other steps can still be run
        button.config(state=tk.DISABLED)
        self.status_var.set(f"Running: {label}")
        self._log(f"\n[RUN] {label}")

        def task():
            try:
                func()
            except Exception as e:
                self._call_on_ui(lambda err=str(e): messagebox.showerror("Error", err))
                self._log(f"[ERROR] {e}")
            finally:
                self._call_on_ui(lambda: button.config(state=tk.NORMAL))
                self._call_on_ui(lambda: self.status_var.set("Idle"))
                self._log(f"[DONE] {label}")

        threading.Thread(target=task, daemon=True).start()

    def _on_close(self):
        """Close app gracefully and stop scheduling UI work from workers."""
        self._closing = True
        self._set_buttons_state(tk.DISABLED)
        self.destroy()

    def _set_buttons_state(self, state):
        for b in self.buttons:
            b.config(state=state)


def main():
    if tk is None:
        print("Tkinter is not available. Running requirement-based core flow in console mode.")
        Runner().step_1_to_4()
        return

    app = MainMenu()
    app.mainloop()


if __name__ == "__main__":
    main()