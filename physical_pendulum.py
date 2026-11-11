
import math
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------- Configuration -------------------

N = 1

masses = np.ones(N)
lengths = np.ones(N)

# Local coordinates:
# +a to the right, +b down when theta = 0
a_cm = np.array([0.40]*1)
b_cm = .5 * lengths

I_cm = masses * lengths**2 / 12.0

initial_thetas = np.full(N, 0)
initial_omegas = np.zeros(N)

ORDER = None
PHYSICAL_PENDULUM = True

# Reset for if not physical but point masses
if PHYSICAL_PENDULUM:
    I_cm = I_cm
    a_cm = a_cm
    b_cm = b_cm
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

START, END, FRAMES = 0.0, 20.0, 1000

G = 9.810

A_0 = 1.0
W_N = 1.0
ZETA = 0.5

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
    return t, np.zeros_like(t, dtype=float)


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

def applied_torques(t, theta, omega):
    return np.zeros(N)


# ------------------- Equations of Motion -------------------

def motion(t, state):
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

    F = applied_torques(t, thetas, theta_dots).copy()

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

def geometry(theta):
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

def relative_energy(state):
    thetas = state[0::2]
    theta_dots = state[1::2]

    A = dd_theta_matrix(
        masses, I_cm, lengths, a_cm, b_cm,
        thetas, None
    )

    T = 0.5 * theta_dots @ A @ theta_dots

    com, _ = geometry(thetas)
    V = G * masses @ com[:, 1]

    return T, V, T + V


def power_balance(t, state):
    thetas = state[0::2]
    theta_dots = state[1::2]

    ax, ay = acceleration(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, t)

    R = np.hypot(a_cm, b_cm)
    phi = np.arctan2(a_cm, b_cm)

    P_pivot = 0.0

    for k in range(N):
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

    P_torque = applied_torques(t, thetas, theta_dots) @ theta_dots

    return P_pivot + P_damping + P_torque


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

t = np.linspace(START, END, FRAMES)

y0 = np.column_stack((
    initial_thetas,
    initial_omegas
)).ravel()

solDOP853 = solve_ivp(
    motion,
    (START, END),
    y0,
    t_eval=t,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12
)
solRK45= solve_ivp(
    motion,
    (START, END),
    y0,
    t_eval=t,
    method="RK45",
    rtol=1e-10,
    atol=1e-12
)
solRK23= solve_ivp(
    motion,
    (START, END),
    y0,
    t_eval=t,
    method="RK23",
    rtol=1e-10,
    atol=1e-12
)
solRadau= solve_ivp(
    motion,
    (START, END),
    y0,
    t_eval=t,
    method="Radau",
    rtol=1e-10,
    atol=1e-12
)
solBDF= solve_ivp(
    motion,
    (START, END),
    y0,
    t_eval=t,
    method="BDF",
    rtol=1e-10,
    atol=1e-12
)

if not solDOP853.success:
    raise RuntimeError(solDOP853.message)

errors = []
def solve(sol):
# ------------------- Pivot and Link Positions -------------------
    xp, yp = pivot_trajectory(USE_EARTHQUAKE, USE_OTHER_ACCELERATION, sol.t)
    
    theta = sol.y[0::2]
    omega = sol.y[1::2]
    
    frames = len(sol.t)
    
    joints = np.zeros((N + 1, 2, frames))
    centers = np.zeros((N, 2, frames))
    
    for f in range(frames):
        com, joint = geometry(theta[:, f])
    
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
        relative_energy(sol.y[:, f])
        for f in range(frames)
    ])
    
    power = np.array([
        power_balance(sol.t[f], sol.y[:, f])
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
    
    residual_percent = 100 * residual / energy_scale
    int_error = np.trapezoid(np.abs(residual_percent),t, 0.5)
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
        residual_percent,
        label="Energy Balance Residual Fraction"
    )
    
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Residual Fraction"
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
    
    
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=lambda: update(0),
        interval=1000 * np.mean(np.diff(sol.t)),
        blit=False,
        repeat=False
    )
    
    fig._animation = animation
    
    plt.show()
solve(solDOP853)
solve(solRK45)
solve(solRK23)
solve(solRadau)
solve(solBDF)
methods = ['DOP853','RK45','RK23','Radau','BDF']
colors = ['red', 'gold', 'blue', 'purple','skyblue']
plt.bar(methods,errors, color = colors)
plt.yscale('log')
plt.ylabel('Total Energy Error')
plt.xlabel('Numerical Integration Methods')
plt.show()