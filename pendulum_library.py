import math
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pypalettes import load_cmap
from pathlib import Path


# Taylor approximation for sin and cos
def trig(theta, order=None):
    """
    Return approximations to sin(theta), cos(theta).

    order=None uses exact NumPy trigonometric functions.

    For integer order, retain polynomial terms up to
    the specified degree.

    QUADRANT_REDUCTION controls whether the argument
    is reduced before Taylor evaluation.
    """
    theta = np.asarray(theta, dtype=float)

    if order is None:
        return np.sin(theta), np.cos(theta)

    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(
            "ORDER must be a positive integer or None."
        )

    # Quadrant reduction
    quadrant = np.rint(
        theta / (np.pi/2)
    ).astype(np.int64)
    r = theta - quadrant*(np.pi/2)

    s = np.zeros_like(r, dtype=float)
    c = np.zeros_like(r, dtype=float)

    for i in range((order + 1)//2):
        s += (
            (-1)**i * r**(2*i + 1)
            / math.factorial(2*i + 1)
        )

    for i in range(order//2 + 1):
        c += (
            (-1)**i * r**(2*i)
            / math.factorial(2*i)
        )

    q = quadrant % 4

    sine = np.select(
        [q == 0, q == 1, q == 2, q == 3],
        [s, c, -s, -c]
    )

    cosine = np.select(
        [q == 0, q == 1, q == 2, q == 3],
        [c, -s, -c, s]
    )

    return sine, cosine

# ------------------- Pivot Acceleration -------------------

def earthquake(t):
    A_0 = 1.0
    W_N = 1.0
    ZETA = 0.5

    t = np.asarray(t)

    if not 0 < ZETA < 1:
        raise ValueError("ZETA must be between zero and one.")

    alpha = ZETA * W_N
    wd = W_N * np.sqrt(1 - ZETA**2)

    ax = A_0 * np.exp(-alpha * t) * (
        np.sin(wd * t)
        - wd / alpha * np.cos(wd * t)
    )

    return ax, np.zeros_like(t, dtype=float)


def other_acceleration(t):
    t = np.asarray(t)
    return -t, np.zeros_like(t, dtype=float)


def acceleration(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, t):
    return earthquake(t) if USE_EARTHQUAKE else other_acceleration(t) if USE_OTHER_ACCELERATION else (0, 0)


def pivot_trajectory(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, t, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0):
    t = np.asarray(t, dtype=float)

    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must be a strictly increasing array.")

    ax, ay = acceleration(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, t)

    ax = np.broadcast_to(ax, t.shape)
    ay = np.broadcast_to(ay, t.shape)

    vx = vx0 + cumulative_trapezoid(ax, t, initial=0.0)
    vy = vy0 + cumulative_trapezoid(ay, t, initial=0.0)

    x = x0 + cumulative_trapezoid(vx, t, initial=0.0)
    y = y0 + cumulative_trapezoid(vy, t, initial=0.0)

    return x, y

# ------------------- Adjusted Length and Angles -------------------

def length_defined(lengths, a_cm, b_cm, index, pendulum_number):
    if index < pendulum_number:
        return lengths[index]
    elif index == pendulum_number:
        distance_to_cm = (a_cm[index] ** 2 + b_cm[index] ** 2) ** (1/2)
        return distance_to_cm
    elif index > pendulum_number:
        return 0

def angle_defined(angles, a_cm, b_cm, index, pendulum_number):
    if index < pendulum_number:
        return angles[index]
    elif index == pendulum_number:
        phi = math.atan2(a_cm[index], b_cm[index])
        return angles[index] + phi
    elif index > pendulum_number:
        return 0

# ------------------- Derived Matrix Equations -------------------

def dd_theta_matrix(masses, I_cm, lengths, a_cm, b_cm, thetas, order):
    N = len(masses)
    matrix = np.zeros((N, N))

    for k in range(N):
        for j in range(N):

            # Rotational inertia about center of mass
            if k == j:
                matrix[k, j] = I_cm[k]

            # Translational inertia contributions
            for n in range(max(k, j), N):
                theta_nk = angle_defined(
                    thetas, a_cm, b_cm, k, n
                )
                theta_nj = angle_defined(
                    thetas, a_cm, b_cm, j, n
                )

                length_nk = length_defined(
                    lengths, a_cm, b_cm, k, n
                )
                length_nj = length_defined(
                    lengths, a_cm, b_cm, j, n
                )

                matrix[k, j] += (
                    masses[n]
                    * length_nk
                    * length_nj
                    * trig(
                        theta_nk - theta_nj, order
                    )[1]
                )

    return matrix

def d_theta_2_matrix(masses, lengths, a_cm, b_cm, thetas, order):
    N = len(masses)
    matrix = np.zeros((N, N))

    for k in range(N):
        for j in range(N):

            for n in range(max(k, j), N):
                theta_nk = angle_defined(
                    thetas, a_cm, b_cm, k, n
                )
                theta_nj = angle_defined(
                    thetas, a_cm, b_cm, j, n
                )

                length_nk = length_defined(
                    lengths, a_cm, b_cm, k, n
                )
                length_nj = length_defined(
                    lengths, a_cm, b_cm, j, n
                )

                matrix[k, j] += (
                    masses[n]
                    * length_nk
                    * length_nj
                    * trig(
                        theta_nk - theta_nj, order
                    )[0]
                )

    return matrix

def theta_matrix(masses, lengths, a_cm, b_cm, thetas, order, t, g = 9.81, use_earthquake=False, use_other_acceleration = False):
    N = len(masses)
    vector = np.zeros(N)

    ax, ay = acceleration(use_earthquake, use_other_acceleration, t)

    for k in range(N):
        for n in range(k, N):
            theta_nk = angle_defined(
                thetas, a_cm, b_cm, k, n
            )

            length_nk = length_defined(
                lengths, a_cm, b_cm, k, n
            )

            vector[k] += (
                masses[n]
                * length_nk
                * ax * trig(theta_nk, order)[1]
                + masses[n]
                * length_nk
                * (ay + g) * trig(theta_nk, order)[0]
            )

    return vector

def applied_torques(N, t, theta, omega):
    return np.zeros(N)

# ------------------- Equations of Motion -------------------

def motion(t, state, masses, I_cm, lengths, a_cm, b_cm,
        thetas, ORDER, G, USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
        USE_DAMPING, DAMPING):
    thetas = state[0::2]
    theta_dots = state[1::2]

    A = dd_theta_matrix(
        masses, I_cm, lengths, a_cm, b_cm,
        thetas, ORDER
    )

    B = d_theta_2_matrix(
        masses, lengths, a_cm, b_cm,
        thetas, ORDER
    )

    gravity = theta_matrix(
        masses, lengths, a_cm, b_cm,
        thetas, ORDER, t, g=G,
        use_earthquake=USE_EARTHQUAKE,
        use_other_acceleration=USE_OTHER_ACCELERATION
    )

    F = applied_torques(len(masses), t, thetas, theta_dots).copy()

    if USE_DAMPING:
        F -= DAMPING * theta_dots

    theta_ddot = np.linalg.solve(
        A,
        F - B @ (theta_dots**2) - gravity
    )

    derivative = np.empty_like(state)
    derivative[0::2] = theta_dots
    derivative[1::2] = theta_ddot

    return derivative

# ------------------- Geometry for Visualization -------------------

def geometry(theta, a_cm, b_cm, lengths):
    n = len(theta)

    s, c = np.sin(theta), np.cos(theta)

    joint = np.zeros((n + 1, 2))
    com = np.zeros((n, 2))

    for k in range(n):
        com[k] = joint[k] + np.array([
            a_cm[k] * c[k] + b_cm[k] * s[k],
            a_cm[k] * s[k] - b_cm[k] * c[k]
        ])

        joint[k + 1] = joint[k] + lengths[k] * np.array([
            s[k], -c[k]
        ])

    return com, joint


# ------------------- Energy -------------------

def relative_energy(state, masses, I_cm, lengths, a_cm, b_cm, G):
    thetas = state[0::2]
    theta_dots = state[1::2]

    A = dd_theta_matrix(
        masses, I_cm, lengths, a_cm, b_cm,
        thetas, None
    )

    T = 0.5 * theta_dots @ A @ theta_dots

    com, _ = geometry(thetas, a_cm, b_cm, lengths)
    V = G * masses @ com[:, 1]

    return T, V, T + V


def power_balance(t, state, masses, lengths, a_cm, b_cm,
        USE_EARTHQUAKE, USE_OTHER_ACCELERATION, USE_DAMPING, DAMPING):
    thetas = state[0::2]
    theta_dots = state[1::2]

    ax, ay = acceleration(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, t)

    R = np.hypot(a_cm, b_cm)
    phi = np.arctan2(a_cm, b_cm)

    P_pivot = 0.0

    for k in range(len(masses)):
        vx = (
            np.sum(
                lengths[:k] * np.cos(thetas[:k])
                * theta_dots[:k]
            )
            + R[k] * np.cos(thetas[k] + phi[k])
            * theta_dots[k]
        )

        vy = (
            np.sum(
                lengths[:k] * np.sin(thetas[:k])
                * theta_dots[:k]
            )
            + R[k] * np.sin(thetas[k] + phi[k])
            * theta_dots[k]
        )

        P_pivot -= masses[k] * (
            float(ax) * vx + float(ay) * vy
        )

    P_damping = (
        -DAMPING * np.dot(theta_dots, theta_dots)
        if USE_DAMPING else 0.0
    )

    P_torque = applied_torques(len(masses), t, thetas, theta_dots) @ theta_dots

    return P_pivot + P_damping + P_torque

def solve(
    sol,
    masses, I_cm, lengths, a_cm, b_cm,
    G, ORDER,
    USE_EARTHQUAKE, USE_OTHER_ACCELERATION,
    USE_DAMPING, DAMPING,
    FOLLOW_PIVOT, TRACE_PATH,
    FRAMERATE, GIF_FRAMERATE, PLAY_BACK,
    errors):
    
    N = len(masses)
    # ------------------- Pivot and Link Positions -------------------
    xp, yp = pivot_trajectory(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, sol.t)
    
    theta = sol.y[0::2]
    omega = sol.y[1::2]
    
    frames = len(sol.t)
    
    joints = np.zeros((N + 1, 2, frames))
    centers = np.zeros((N, 2, frames))
    
    for f in range(frames):
        com, joint = geometry(theta[:, f], a_cm, b_cm, lengths)
    
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
        relative_energy(sol.y[:, f], masses, I_cm, lengths, a_cm, b_cm, G)
        for f in range(frames)
    ])
    
    power = np.array([
        power_balance(sol.t[f], sol.y[:, f], masses, lengths, a_cm, b_cm,
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


def solver_accuracy(methods, angular_errors, errors, solver_runtimes, function_evaluations, OUTPUT_FOLDER):
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

    # ----
    
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
