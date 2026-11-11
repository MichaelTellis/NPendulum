import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp




'''
Physics---------------------------------------

theta_i = angle of pendulum i from vertical (rad) (absolute angle)
omega_i = theta_i'

Each pendulum in the chain has (see Pendulum class below):
    m     mass
    L     length distance between top and bottom pivots
    cm     distance from its top pivot to its  center of mass
    I_cm  moment of inertia about its  center of mass
    b     damping at the top joint: damping term = -b * (omega_i - omega_{i-1})

Lagrangian mechanics: L = T-V
T = 1/2 sum_ij M_ij omega_i omega_j

V = g sum_i B_i (1 - cos theta_i)

Note link 0 is the top link:
link_m_ij = distance pendulum i's center of mass moves per radian of theta_j.
    link_m_ij = 0 if j > i,  movement of pendulums below i does not affect pendulum i's center of mass.
    link_m_ii = cm_i, center of mass of pendulum i moves by its own distance to pivot above per radian of theta_i.
    link_m_ij = L_j for j < i, Since pendulum j is above i, it impacts cm_i's motion. L_j is the distance from the pivot of pendulum j to the pivot of pendulum j+1. The center of mass of pendulum i moves by L_j per radian of theta_j.

M_ij is the inertia matrix of the system, taking account the angles of each pendulum. 
The:  konecker_delta_ij * I_cm_i ; represents the moment of inertia of pendulum i about its center of mass.
The:  A_ij * cos(theta_i - theta_j) ; Measures the coupling between pendulum i and pendulum j. When their angles line up, the more they drag on each other.

M_ij = A_ij * cos(theta_i - theta_j) + konecker_delta_ij * I_cm_i

These matrices are constant over time, but depend on the parameters of the pendulums in the chain.
A is the moment of inertia matrix. For diagonal element A_ii = sum_k m_k * link_m_ki^2 which is the moment of inertia of rod k about its pivot + the translational inertia of all the rods below it. 
For off diagonal elements A_ij = sum_k m_k * link_m_ki * link_m_kj. These terms measure how much rod j drags on rod i. accelerating rod j produces a torque on joint i through the mass they share.

A_ij = sum_k m_k * link_m_ki * link_m_kj 

B_i represents the torque on pendulum i due to gravity. It is the sum of mass for each pivot below pendulum i * length of pendulum i + m_i*cm_i. 
This treats the combined mass of all the pendulums below i as a single point mass at the end of pendulum i.

B_i = sum_k m_k * link_m_ki

Lagrange's equations of motion for the system result in:
sum_j M_ij theta_j'' = - sum_j A_ij sin(theta_i - theta_j) omega_j^2 - g b_i sin(theta_i) + Q_i

AI is helping me with these explanations below cause I don't fully understand them I'm gonna be honest:

Q_i is a damping torque at joint i. Such as friction
Q_i = 0, energy is conserved

sum_j M_ij theta_j'' is the inertia term. Think I*alpha in the torque equation, so it represents the tangential acceleration
The off-diagonal terms say that when another rod j speeds up, the shared masses get shoved, and joint i feels it
the cos(theta_i - theta_j) term says that if the two rods are aligned, they drag on each other more than if they are perpendicular.

sum_j A_ij sin(theta_i - theta_j) omega_j^2 is the centripital term. Rod j spins at omega_j, so it will pull the masses below it inward.
A pull along rod j can still create a torque about joint i if rods i and j aren't lined up. 
The sin(theta_i - theta_j) is how much of that pull points perpendicular to rod i. 

g b_i sin(theta_i) is the gravitational torque on rod i. 
The sin(theta_i) term says that if the rod is vertical, gravity doesn't pull it to one side or the other. 
The b_i term is the combined mass of all the rods below rod i + rod i, so it measures how much torque gravity exerts on rod i.

For each time step, the simulation solves for theta''. 
Therefore, we need to know the state of the RHS of the equation, and the inertia matrix M.
Solving for acceleration allows (solve_ivp) to integrate the system forward in time.

'''

#CODE-----------------------------------------
class Pendulum:
    def __init__(self, m, L, cm ,I_cm , b = 0):
        self.m = m        # kg
        self.L = L        # m, length of one pendulum
        self.cm = cm  # m, distance from pivot to center of mass
        self.I_cm = I_cm  # kg*m^2, moment of inertia about center of mass
        self.b = b        # damping at pivot (set to 0 i haven't coded damping yet)

def uniform_massive_rod(m=1.0, L= 1.0, b= 0.0):
    return Pendulum(m=m, L=L, cm=L/2, I_cm=m*L**2/12, b=b) 

class pendulumChain:
    def __init__(self, pendulums: list, g=9.81):
        '''
        pendulums: list-like of Pendulum objects
        g: float
            gravitational acceleration
        '''
        self.pendulums = list(pendulums)
        self.n = len(pendulums)
        self.g = g

        self.m = np.array([p.m for p in pendulums])
        self.L = np.array([p.L for p in pendulums])
        self.cm = np.array([p.cm for p in pendulums])
        self.I_cm = np.array([p.I_cm for p in pendulums])
        self.b = np.array([p.b for p in pendulums])

        #link matrix: state[k,i]  = motion of link k's center of mass per radian of theta_i
        link_m =np.zeros((self.n,self.n))
        #Fill in link matrix. Diagonal elements are equal to the distance from the pivot of link i to its center of mass.
        #Element link_m[k,i] = length of link i
        for i in range(self.n):
            link_m[i,:i] = self.L[:i]
            link_m[i,i] = self.cm[i]
        self.link_m = link_m
        #I asked AI for help with this part. Matrix A is supposed to be the Moment of Inertia matrix for the system. 
        # A_ij = sum_k m_k ell_ki ell_kj
        self.A = link_m.T @ np.diag(self.m) @ link_m
        # B_i  = sum_k m_k ell_ki
        self.B = link_m.T @ self.m
        #AI said to precalculate thse since they are constant over time to make my code faster.
        self.I_diag = np.diag(self.I_cm) # equals to kronecker delta of moments of inertia
        self.g_b = self.g * self.B 

        #no damping because iam tired

    def state(self,t,y):
        '''
        Right-hand side for solve_ivp: y = [theta_0..theta_{n-1}, omega_0..omega_{n-1}].
        '''
        # extract out prev theta and omega states
        theta = y[:self.n]
        omega = y[self.n:]
        # pairwise differences of thetas
        diff =np.subtract.outer(theta, theta)

        #Calculate M matrix (see physics section)
        M = self.A*np.cos(diff) + self.I_diag

        # Caclulate rhs vector (see physics section)
        rhs = -((self.A * np.sin(diff)) @ (omega * omega)) - self.g_b * np.sin(theta)


        result = np.empty(2*self.n)
        # d theta /dt = omega
        result[:self.n] = omega
        # Solve the matrix equation to find theta'', aka d omega /dt.
        #Solve Ax = b, where A = M and b = rhs, x = theta''.
        result[self.n:] = np.linalg.solve(M,rhs)

        return result

    def joint_pos(self, theta):
        '''
        theta: (n,) or (n,T) array. (n,) for one time instance, (n,T) for a timespan.

        calculates the x,y position for every joint. joint 0 (row 0)is the fixed pivot at the top,
        joint n (row n) is the bottom of the last pendulum. returns position for every time step.
        Therefore it returns two 2D arrays of shape (n+1, T), where T is the number of time steps.

        x_i = (sum to n = i: L_n * sin(theta_n))
        y_i = (sum to n = i: L_n * -cos(theta_n))
    
        '''
        # Converting theta into matrix if not already. -1 in column so numpy figures out on its own how many columns are needed
        theta_reshape = np.reshape(theta, (self.n,-1))

        zero_row = np.zeros((1,theta_reshape.shape[1]))
        
        x = np.vstack([zero_row, np.cumsum(self.L[:,None] * np.sin(theta_reshape),axis = 0)])
        y = np.vstack([zero_row, np.cumsum(self.L[:,None] * -np.cos(theta_reshape),axis = 0)])
        return x, y


    def energy(self, theta, omega):
        '''
            calculates energy of the system at a given time.
            theta, omega: arrays of shape (n,)
            E_N = K + U
            E_N = (1/2 omega^T * M * omega ) + (g * sum_i B_i * (1 - cos_taylor(theta_i)))
    
            M_ij = A_ij taylor_cos(theta_i - theta_j) + kronecker_delta_ij * I_cm_i
            
        '''
        diff = np.subtract.outer(theta,theta)
        M = self.A * np.cos(diff) + self.I_diag
        KE = 0.5*(omega @ M @ omega) 
        PE = self.g_b * (1- np.cos(theta))
        total_E =  np.sum(KE)+ np.sum(PE)
        
        return total_E
    

def simulate(chain,theta0,omega0=0.0,t_span = (0,10), num_points = 1000, method = "LSODA"):
    '''
    chain: PendulumChain obj
    theta0: float or array of len n. 
        Starting angle of each link in radians, measured from straight down.
    omega0: float or array of len n.
        Starting angular velocity of each link in rad/s.
    t_span: (float, float)
        (start,end) times in sec
    num_points: int
        number of times at which the state is saved
    method: str
        Integration method for solve_ivp. I use "LSODA" because AI said it was fast for this case

    returns:
    sol: scipy OdeResult
        sol.t: array shape (num_points,)
            times saved
        sol.y: array shape (2n, num_points)
            angles: rows 0 to n-1: sol.y[:n]
            angular velocities: rows n to 2n -1: sol.y[n:]
    '''
    theta0 = np.broadcast_to(np.array(theta0),(chain.n,))
    omega0 = np.broadcast_to(np.array(omega0),(chain.n,))
    y0 = np.concatenate([theta0,omega0])
    t = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(chain.state,t_span,y0,method=method,t_eval = t,rtol=1e-10,atol=1e-10)
    return sol

#plotting routine
def plotting_routine(chain,sol,save_as=None):
    '''
    Plots the theta vs time graph for each pendulum in the chain. You can save the graph as well

    chain: PendulumChain obj
    sol: Scipy OdeResult
    save_as: str
        file format to save as
    '''
    theta,omega = sol.y[:chain.n], sol.y[chain.n:]
    fig, (ax_theta) = plt.subplots(1,1, figsize=(6,6))
    for i in range(chain.n):
        ax_theta.plot(sol.t,theta[i],label = f"link {i + 1}")
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("Theta (Rad)")
    ax_theta.set_title("Link angles vs time")
    ax_theta.legend()



    if save_as:
        fig.savefig(save_as, dpi=150)
    #plt.show() 
    #I commented out plt.show() because I wanted running the plotting routine multiple times to display them all at once

    
def animate(chain,sol,step = 1,save_as =None, title=None):
    theta,omega = sol.y[:chain.n], sol.y[chain.n:]
    x,y  = chain.joint_pos(theta)
    dt = sol.t[1] - sol.t[0]
    fps = 1 / (dt * step)  
    fig, (ax_time,ax_pos) = plt.subplots(1,2,figsize= (14,7))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    ax_pos.set_xlim(-1.25*np.sum(chain.L),1.25*np.sum(chain.L))
    ax_pos.set_ylim(-1.25*np.sum(chain.L),1.25*np.sum(chain.L))
    ax_pos.set_aspect("equal")
    path, = ax_pos.plot(0,0, color='red', alpha=0.2)
    points, = ax_pos.plot(0,0,marker='o', markerfacecolor='r', color='k')
    ax_pos.set_xlabel("x (m)")
    ax_pos.set_ylabel("y (m)")
    ax_pos.set_title(f"{chain.n}-link physical pendulum")

    ax_time.set_xlim(sol.t[0], sol.t[-1])
    ax_time.set_ylim(np.min(theta) - 0.1, np.max(theta) + 0.1)
    theta_paths = [ax_time.plot(0,0,label = f" pendulum {i+1}")[0] for i in range(chain.n)]
    time_text = ax_pos.text(0.03, 0.95, "", transform=ax_pos.transAxes)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Theta (Rad)")
    ax_time.set_title("Link angles vs time")
    ax_time.legend()
    fig.tight_layout()



    def update(f):
        path.set_data(x[-1,:f+1],y[-1,:f+1])
        points.set_data(x[:,f],y[:,f])
        for i in range(chain.n):
            theta_paths[i].set_data(sol.t[:f+1],theta[i,:f+1])
        time_text.set_text(f"t = {sol.t[f]:.2f}s")
        return [path,points, *theta_paths,time_text]

    animation = FuncAnimation(fig,update,frames=range(0,len(sol.t),step), interval = 1000/fps,blit=True)
    if save_as:
        writer = "ffmpeg" if save_as.endswith(".mp4") else "pillow"
        animation.save(save_as, writer=writer, fps=fps)
    return animation
    
    
    
'''
ex: 
theta0 = np.pi/2
omega0=0.3

pendulum = uniform_massive_rod(m=1,L=1,b=0)
chain = pendulumChain([pendulum,pendulum,pendulum],g=9.81)

sol = simulate(chain,theta0,omega0)
plotting_routine(chain,sol)
animate(chain,sol)
'''


