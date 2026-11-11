import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
'''
T = I*alpha:

I * theta'' = -m g d sin(theta)
theta'' = -(m g d / I) sin(theta)

Integrate two first order ODE to solve phase space and time evolution of physical pendulum:
theta' = omega
omega' = - (m g d/I) sin(theta) - (b/I) * omega

'''

#Parameters
class PendulumParams:
    def __init__(self, g=9.81, m=1.0, L=1.0, b=0.01):
        self.g = g        # m/s^2
        self.m = m        # kg
        self.L = L        # m
        self.b = b        # damping

        self.d = L / 2
        self.I = m * L**2 / 3
        self.omega0_sq = m * g * self.d / self.I


#initial conditions
theta0 = np.pi/4  # init angle in radians
omega0 = 0.0      # init angular velocity in rad/s
t_span = [0, 10]  # time span for sim

#differential equation for uniform physical pendulum that solve_ivp solves
def uniform_physical_pendulum_analytic(t,y,p=PendulumParams()):
    theta , omega = y
    return [omega, - p.omega0_sq * np.sin(theta) - p.b/p.I * omega]
# energy of the physical pendulum at given theta and omega
def energy(theta, omega, p=PendulumParams()):
    return 0.5 * p.I * omega**2 + p.m * p.g * p.d * (1 - np.cos(theta))

#runs simulation with given intital conditions
def simulate_physical_pendulum(theta0: float, omega0: float, t_span: list, num_points=1000, p=PendulumParams()):
    y0 = [theta0, omega0]
    t = np.linspace(t_span[0], t_span[1], num_points)
    #solve the ODE using solve_ivp
    return solve_ivp(uniform_physical_pendulum_analytic,t_span, y0, method = 'RK45', args = (p,), t_eval=t,rtol=1e-10, atol=1e-10)


#plot results of simulation.
def plotting_routine(sol):
    fig, (ax_time, ax_phase, ax_energy) = plt.subplots(3, 1, figsize=(10, 10))
    #time plot
    theta, omega = sol.y
    ax_time.plot(sol.t, theta, label='Theta (rad)')
    ax_time.plot(sol.t, omega, label='Omega (rad/s)')
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Vars')
    ax_time.set_title('Theta and Omega vs Time')
    ax_time.legend()
    #phase plot
    ax_phase.plot(theta, omega)
    ax_phase.set_xlabel('Theta (rad)')
    ax_phase.set_ylabel('Omega (rad/s)')
    ax_phase.set_title('Phase Space Plot')
    #energy plot to check energy conservation
    ax_energy.plot(sol.t, energy(theta, omega))
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy (J)')
    ax_energy.set_title('Energy vs Time')

    plt.tight_layout()
    fig.savefig("pendulum.png", dpi=150)
    plt.show()

def sol_animate(sol, p=PendulumParams()):
    theta, omega = sol.y
    step = 1
    L = p.L
    frames = range(0, len(sol.t), step)

    fig_time,  ax_time  = plt.subplots(figsize=(6, 6))
    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
    fig_pend,  ax_pend  = plt.subplots(figsize=(6, 6))
    ax_time.update_datalim(np.column_stack([sol.t, theta]))
    ax_phase.update_datalim(np.column_stack([theta, omega]))
    ax_pend.set_xlim(-1.5*L, 1.5*L)
    ax_pend.set_ylim(-1.5*L, 1.5*L)
    time_point = ax_time.plot(0, 0, marker='o', markerfacecolor='r', color='k')[0]
    time_path = ax_time.plot(0, 0, color='red', alpha=0.2)[0]
    time_anim = FuncAnimation(fig_time, func_time, frames=frames, fargs=(time_point, time_path, sol), interval=1000/60, blit=True)
    phase_point = ax_phase.plot(0, 0, marker='o', markerfacecolor='r', color='k')[0]
    phase_path = ax_phase.plot(0, 0, color='red', alpha=0.2)[0]
    phase_anim = FuncAnimation(fig_phase, func_phase, frames=frames, fargs=(phase_point, phase_path, sol), interval=1000/60, blit=True)
    pend_point = ax_pend.plot(0, 0, marker='o', markerfacecolor='r', color='k')[0]
    pend_path = ax_pend.plot(0, 0, color='red', alpha=0.2)[0]
    pend_anim = FuncAnimation(fig_pend, func_pend, frames=frames, fargs=(pend_point, pend_path, sol, L), interval=1000/60, blit=True)

    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Theta (rad) and Omega (rad/s)')
    ax_time.set_title('Theta and Omega vs Time')
    ax_phase.set_xlabel('Theta (rad)')
    ax_phase.set_ylabel('Omega (rad/s)')
    ax_phase.set_title('Phase Space Plot')
    ax_pend.set_xlabel('X Position (m)')
    ax_pend.set_ylabel('Y Position (m)')
    ax_pend.set_title('Pendulum Animation')
    '''
    time_anim.save('time_animation.gif', writer='pillow', fps=60)
    phase_anim.save('phase_animation.gif', writer='pillow', fps=60)
    pend_anim.save('pendulum_animation.gif', writer='pillow', fps=60)
    '''
    

    plt.show()




def func_pend(i, point, path, sol,L):
    theta, omega = sol.y
    point.set_data([0, L*np.sin(theta[i])], [0, -L*np.cos(theta[i])])
    path.set_data(L*np.sin(theta[:i]), -L*np.cos(theta[:i]))
    return point, path

def func_time(i, point, path, sol):
    theta, omega = sol.y
    point.set_data([sol.t[i]], [theta[i]])
    path.set_data([sol.t[:i]], [theta[:i]])
    return point, path
def func_phase(i, point, path, sol):
    theta, omega = sol.y
    point.set_data([theta[i]], [omega[i]])
    path.set_data([theta[:i]], [omega[:i]])
    return point, path
#sol = simulate_physical_pendulum(theta0, omega0, t_span)
#sol_animate(sol)
