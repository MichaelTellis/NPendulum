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