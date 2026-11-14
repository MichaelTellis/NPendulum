
import math
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib
PLOT_SHOW = False
if not PLOT_SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pypalettes import load_cmap
from pathlib import Path
import pendulum_library as pendulum

# ------------------- Configuration -------------------

N = 4

masses = np.ones(N)
lengths = np.ones(N)

initial_thetas = np.full(N, 0)
initial_omegas = np.zeros(N)

ORDER = None
PHYSICAL_PENDULUM = False

# Reset for if not physical but point masses
if PHYSICAL_PENDULUM:
    # Local coordinates: +a to the right, +b down when theta = 0
    I_cm = masses * lengths**2 / 12.0
    a_cm = np.array([0.40]*N)
    b_cm = .5 * lengths
else:
    I_cm = np.zeros(N)
    a_cm = np.zeros(N)
    b_cm = lengths


USE_EARTHQUAKE = False
USE_OTHER_ACCELERATION = True # ignored if USE_EARTHQUAKE = True
USE_DAMPING = True
DAMPING = 0.3

FOLLOW_PIVOT = True
TRACE_PATH = True
GIF_FRAMERATE = 40
SAVE_IMAGES = True
RUN_ALL = True # runs all integration techniques

START, END, FRAMERATE, PLAY_BACK = 0, 60, 60, 0.5
RUNTIME = END - START
OUTPUT_FOLDER = Path('pendulum_output')

G = 9.810

# ------------------- Validation -------------------

for arr in (
    masses, lengths, a_cm, b_cm,
    I_cm, initial_thetas, initial_omegas
):
    if np.shape(arr) != (N,):
        raise ValueError("All per-link arrays must have length N.")

if np.any(masses <= 0) or np.any(lengths <= 0) or np.any(I_cm < 0):
    raise ValueError("Invalid masses, lengths, or inertias.")


# ------------------- Numerical Integration -------------------

t = np.linspace(START, END, int(RUNTIME*FRAMERATE))

y0 = np.column_stack((
    initial_thetas,
    initial_omegas
)).ravel()

methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
methods_one = 'RK45'

solutions = {}
solver_runtimes = {}

motion_args = (
    masses, I_cm, lengths, a_cm, b_cm,
    initial_thetas, ORDER, G,
    USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
    USE_DAMPING, DAMPING
)

if RUN_ALL:
    for m in methods:
        start_time = time.perf_counter()
        solutions[m] = solve_ivp(
            pendulum.motion, (START, END), y0, t_eval=t,
            method=m, rtol=1e-10, atol=1e-12, args=motion_args
        )
        solver_runtimes[m] = time.perf_counter() - start_time
else:
    start_time = time.perf_counter()
    solutions[methods_one] = solve_ivp(
        pendulum.motion, (START, END), y0, t_eval=t,
        method=methods_one, rtol=1e-10, atol=1e-12, args=motion_args
    )
    solver_runtimes[methods_one] = time.perf_counter() - start_time


# ------------------- High-Accuracy Reference -------------------

if RUN_ALL:
    reference = solve_ivp(
        pendulum.motion, (START, END), y0, t_eval=t,
        method="DOP853", rtol=1e-12, atol=1e-14,
        args=motion_args
    )
    if not reference.success:
        raise RuntimeError(f"Reference integration failed: {reference.message}")
    theta_reference = reference.y[0::2]

# ------------------- Creating Plots -------------------

angular_errors = []
function_evaluations = []

errors = []

for method, solution in solutions.items():
    print(f"Processing {method}")

    if not solution.success:
        print(f"{method} failed: {solution.message}")
        angular_errors.append(np.nan)
        errors.append(np.nan)
        function_evaluations.append(solution.nfev)
        continue

    method_folder = OUTPUT_FOLDER / method
    method_folder.mkdir(parents=True, exist_ok=True)

    function_evaluations.append(solution.nfev)

    if RUN_ALL:
        angle_difference = np.arctan2(
            np.sin(solution.y[0::2] - theta_reference),
            np.cos(solution.y[0::2] - theta_reference)
        )
        mean_squared_error = np.mean(angle_difference**2, axis=0)
        angular_errors.append(np.sqrt(
            np.trapezoid(mean_squared_error, x=solution.t)
            / (solution.t[-1] - solution.t[0])
        ))

    fig, fig_angles, fig_energy, fig_x, animation = pendulum.solve(solution, masses, I_cm, lengths, a_cm, b_cm,
                                                        G, ORDER,
                                                        USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
                                                        USE_DAMPING, DAMPING,
                                                        FOLLOW_PIVOT, TRACE_PATH,
                                                        FRAMERATE, GIF_FRAMERATE, PLAY_BACK,
                                                        errors)

    if SAVE_IMAGES:
        # Save all static plots
        fig_angles.savefig(
            method_folder / "angles.png",
            dpi=300, bbox_inches="tight"
        )

        fig_energy.savefig(
            method_folder / "energy.png",
            dpi=300, bbox_inches="tight"
        )

        fig_x.savefig(
            method_folder / "x_position.png",
            dpi=300, bbox_inches="tight"
        )

        # Save pendulum animation as GIF
        print(f"Saving {method} GIF...", flush=True)
        animation.save(
            method_folder / "pendulum.gif",
            writer="pillow",
            fps=FRAMERATE * PLAY_BACK / max(1, round(FRAMERATE / GIF_FRAMERATE)),
            dpi=80
        )

        print(f"Finished {method} GIF.", flush=True)
    if PLOT_SHOW:
        plt.show()

    plt.close('all')


# ------------------- Solver Accuracy Comparison -------------------
if RUN_ALL: pendulum.solver_accuracy(methods, angular_errors, errors, solver_runtimes, function_evaluations, OUTPUT_FOLDER)

if PLOT_SHOW:
    plt.show()
plt.close('all')