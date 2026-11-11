from OnePhysicalPendulumSim import *
import numpy as np

theta0 = np.pi / 4  # Initial angle (radians)
omega0 = 0        # Initial angular velocity (rad/s)
t_span = [0, 10]    # Time span for simulation (seconds)
params = PendulumParams(g=9.81, m=1.0, L=1.0, b=0.2)  
sol = simulate_physical_pendulum(theta0, omega0, t_span, num_points=1000, p=params)
sol_animate(sol, p=params)