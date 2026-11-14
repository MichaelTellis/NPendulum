
import math
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib
PLOT_SHOW = True
if not PLOT_SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pypalettes import load_cmap
from pathlib import Path
import pendulum_library as pendulum

# ------------------- Configuration -------------------

N = 2

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
GIF_FRAMERATE = 15
SAVE_IMAGES = False
RUN_ONE = True

START, END, FRAMERATE, PLAY_BACK = 0, 60, 60, 0.5
RUNTIME = END - START
OUTPUT_FOLDER = Path('pendulum_output')

G = 9.810

A_0 = 1.0
W_N = 1.0
ZETA = 0.5

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

solutions = {}
solver_runtimes = {}

motion_args = (
    masses, I_cm, lengths, a_cm, b_cm,
    initial_thetas, ORDER, G,
    USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
    USE_DAMPING, DAMPING
)
for m in methods:
    start_time = time.perf_counter()
    solutions[m] = solve_ivp(
        pendulum.motion, (START, END), y0, t_eval=t,
        method=m, rtol=1e-10, atol=1e-12, args=motion_args
    )
    solver_runtimes[m] = time.perf_counter() - start_time

angular_errors = []
function_evaluations = []

errors = []

def solve(sol):
# ------------------- Pivot and Link Positions -------------------
    xp, yp = pendulum.pivot_trajectory(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, sol.t)
    
    theta = sol.y[0::2]
    omega = sol.y[1::2]
    
    frames = len(sol.t)
    
    joints = np.zeros((N + 1, 2, frames))
    centers = np.zeros((N, 2, frames))
    
    for f in range(frames):
        com, joint = pendulum.geometry(theta[:, f], a_cm, b_cm, lengths)
    
        centers[:, :, f] = com + np.array([xp[f], yp[f]])
        joints[:, :, f] = joint + np.array([xp[f], yp[f]])
    
    
    # ------------------- Angle Plot -------------------
    
    fig_angles, ax_angles = plt.subplots(figsize=(9, 4))
    
    for k in range(N):
        ax_angles.plot(sol.t, theta[k], label=f"Link {k+1}")
    
    ax_angles.set(
        xlabel="Time (s)",
        ylabel="Angle (rad)",
        title="Link Angles"
    )
    
    ax_angles.legend()
    ax_angles.grid(True)
    fig_angles.tight_layout()
    
    
    # ------------------- Energy Plot -------------------
    
    E = np.array([
        pendulum.relative_energy(sol.y[:, f], masses, I_cm, lengths, a_cm, b_cm, G)
        for f in range(frames)
    ])
    
    power = np.array([
        pendulum.power_balance(sol.t[f], sol.y[:, f], masses, lengths, a_cm, b_cm,
        USE_EARTHQUAKE, USE_OTHER_ACCELERATION, USE_DAMPING, DAMPING)
        for f in range(frames)
    ])
    
    work = cumulative_trapezoid(
        power, sol.t, initial=0.0
    )
    
    residual = E[:, 2] - E[0, 2] - work
    
    energy_scale = max(
        np.max(np.abs(E[:, 0])),
        np.max(np.abs(E[:, 1])),
        np.max(np.abs(work)),
        1e-12
    )
    
    residual_fraction = residual / energy_scale
    int_error = np.trapezoid(np.abs(residual_fraction), x=sol.t) / (sol.t[-1] - sol.t[0])
    errors.append(int_error)
    
    fig_energy, axes = plt.subplots(
        3, 1, figsize=(9, 7), sharex=True
    )
    
    axes[0].plot(
        sol.t,
        E[:, 2] - E[0, 2],
        label="Measured Relative Energy Change"
    )
    
    axes[0].plot(
        sol.t,
        work,
        "--",
        label="Integrated External Work"
    )
    
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylabel("Energy (J)")
    
    axes[1].plot(
        sol.t,
        residual,
        label="Energy Balance Residual"
    )
    
    axes[1].set(
        ylabel="Residual (J)"
    )
    
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(
        sol.t,
        residual_fraction,
        label="Energy Balance Residual Fraction"
    )
    
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Residual (Fraction)"
    )
    
    axes[2].legend()
    axes[2].grid(True)
    
    fig_energy.tight_layout()
    
    print(
        "Maximum energy balance residual (J):",
        np.max(np.abs(residual))
    )
    
    if ORDER is not None:
        print(
            "Note: finite Taylor order may not preserve "
            "exact Lagrangian energy identities."
        )
    
    
    # ------------------- Animation -------------------
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="Physical Multi-Link Pendulum"
    )
    
    margin = 1.3 * (
        np.sum(lengths) + np.max(np.hypot(a_cm, b_cm))
    )
    
    if not FOLLOW_PIVOT:
        ax.set_xlim(np.min(xp) - margin, np.max(xp) + margin)
        ax.set_ylim(np.min(yp) - margin, np.max(yp) + margin)
    
    rods, = ax.plot(
        [], [], "o-",
        color="black",
        lw=2,
        ms=4
    )
    
    cms = ax.scatter(
        [], [],
        s=65,
        color="tab:red",
        label="Centers of Mass",
        zorder=4
    )
    
    pivot, = ax.plot([], [], "ko", ms=8)
    
    trace, = ax.plot(
        [], [],
        color="tab:red",
        alpha=0.35
    )
    
    clock = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes
    )
    
    ax.legend(loc="upper right")
    
    
    def update(f):
        rods.set_data(
            joints[:, 0, f],
            joints[:, 1, f]
        )
    
        cms.set_offsets(centers[:, :, f])
    
        pivot.set_data(
            [xp[f]],
            [yp[f]]
        )
    
        clock.set_text(f"t = {sol.t[f]:.2f} s")
    
        if TRACE_PATH:
            trace.set_data(
                centers[-1, 0, :f+1],
                centers[-1, 1, :f+1]
            )
    
        if FOLLOW_PIVOT:
            ax.set_xlim(xp[f] - margin, xp[f] + margin)
            ax.set_ylim(yp[f] - margin, yp[f] + margin)
    
        return rods, cms, pivot, trace, clock
    
    # Only change graph if pivot accelerates
    if USE_EARTHQUAKE and USE_OTHER_ACCELERATION == False:
        blit = True
    else:
        blit = False    

    animation = FuncAnimation(
        fig,
        update,
        frames=range(0, frames, max(1, round(FRAMERATE / GIF_FRAMERATE))),
        init_func=lambda: update(0),
        blit=blit,
        interval=1000 * max(1, round(FRAMERATE / GIF_FRAMERATE)) / FRAMERATE / PLAY_BACK,
        repeat=False
    )

    fig._animation = animation
    
    # ----- Plot for x_movement over time - goal of minimizing for earthquake

    x_bottom = centers[-1, 0, :]

    fig_x, ax_x = plt.subplots(figsize=(9, 4))

    ax_x.plot(sol.t, x_bottom - xp, label="Bottom Pendulum - Top Pendulum")

    ax_x.set(
        xlabel="Time (s)",
        ylabel="Relative Horizontal Position (m)",
        title="Bottom Pendulum Horizontal Position Relative to Top Pendulum"
    )

    ax_x.legend()
    ax_x.grid(True)
    fig_x.tight_layout()

    return fig, fig_angles, fig_energy, fig_x, animation

for method, solution in solutions.items():
    print(f"Processing {method}")

    if not solution.success:
        print(f"{method} failed: {solution.message}")
        angular_errors.append(np.nan)
        function_evaluations.append(solution.nfev)
        continue

    method_folder = OUTPUT_FOLDER / method
    method_folder.mkdir(parents=True, exist_ok=True)

    function_evaluations.append(solution.nfev)

    fig, fig_angles, fig_energy, fig_x, animation = solve(solution)

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

cmap = load_cmap("Sunset", keep_first_n=len(methods))
hex_colors = cmap.colors

fig_bar, ax_bar = plt.subplots(figsize=(9, 4))

ax_bar.bar(methods, errors, color=hex_colors)
ax_bar.set_yscale("log")
ax_bar.set_ylabel("Mean Absolute Energy Balance Residual (Fraction)")
ax_bar.set_xlabel("Numerical Integration Methods")

fig_bar.tight_layout()

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

fig_bar.savefig(
    OUTPUT_FOLDER / "solver_comparison.png",
    dpi=300, bbox_inches="tight"
)

if PLOT_SHOW:
    plt.show()

plt.close(fig_bar)

# ------------------- Solver Accuracy Comparison -------------------
# Energy residuals above test energy balance; this chart compares trajectories.
fig_accuracy, ax_accuracy = plt.subplots(figsize=(9, 4))
ax_accuracy.bar(methods, angular_errors, color=hex_colors)
ax_accuracy.set_yscale("log")
ax_accuracy.set(
    xlabel="Numerical Integration Methods",
    ylabel="RMS Angular Difference (rad)",
    title="Trajectory Difference from High-Accuracy DOP853 Reference"
)
ax_accuracy.grid(axis="y", alpha=0.3)
fig_accuracy.tight_layout()
fig_accuracy.savefig(OUTPUT_FOLDER / "trajectory_accuracy.png", dpi=300, bbox_inches="tight")

fig_cost, ax_cost = plt.subplots(figsize=(9, 4))
ax_cost.bar(methods, [solver_runtimes[m] for m in methods], color=hex_colors)
ax_cost.set(
    xlabel="Numerical Integration Methods",
    ylabel="Integration Runtime (s)",
    title="Numerical Solver Runtime (Excluding Plots and GIFs)"
)
ax_cost.grid(axis="y", alpha=0.3)
fig_cost.tight_layout()
fig_cost.savefig(OUTPUT_FOLDER / "solver_runtime.png", dpi=300, bbox_inches="tight")

print("\nMethod | RMS angle difference (rad) | Runtime (s) | Function evaluations")
for m, error, nfev in zip(methods, angular_errors, function_evaluations):
    print(f"{m:6s} | {error:.6e} | {solver_runtimes[m]:.3f} | {nfev}")

if PLOT_SHOW:
    plt.show()
plt.close('all')