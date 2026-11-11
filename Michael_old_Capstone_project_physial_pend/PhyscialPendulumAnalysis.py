import numpy as np
import matplotlib.pyplot as plt
from NPhysicsPendulumSim import pendulumChain, uniform_massive_rod, simulate, plotting_routine, animate
from TaylorPhysicalPendulumSim import TaylorChain

def angle_energy_diff(pendulums, theta0s,omega0s,t_span=(0,10), num_points= 1000,g=9.81, orders = [1,3,5]):
    """
    
    Pendulums: List of Pendulum objects which you will be simulating
    theta0s: 1D array of the set of intitial theta where you want each chain to start.
        You might be able to make it a 2D array where the columns specify the individual starting angles of each pendulum in the chain, not sure
    omega0s: 1D array of the set of initial omegas where you want each chain to start. 

    The analysis will run angle difference comparisons on every pairwise combination of theta0s and omega0s.
    """
    theta0s = np.array(theta0s)
    omega0s = np.array(omega0s)
    print(omega0s)
    num_pends = len(pendulums)
    


    
    #theta0s_2D = np.broadcast_to(theta0s, (theta0s.size,omega0s.size))
    #omega0s_2D = np.broadcast_to(omega0s, (theta0s.size,omega0s.size))
    exact_chain = {"Exact Solution": pendulumChain(pendulums,g=g)}
    chains = {f"Taylor Approx N = {order}": TaylorChain(pendulums,order=order,g=g) for order in orders}


    #y0s = np.stack((theta0s_2D,omega0s_2D),axis = -1)
    #print(y0s)
    theta_diffs= np.zeros((theta0s.size,omega0s.size,len(orders)))
    thetas_exact = np.zeros((theta0s.size,omega0s.size))
    omegas_diffs= np.zeros((theta0s.size,omega0s.size,len(orders)))
    omegas_exact = np.zeros((theta0s.size,omega0s.size))
    energy_diffs = np.zeros((theta0s.size,omega0s.size,len(orders)))
    energy_exact = np.zeros((theta0s.size,omega0s.size))
    
    for i, theta0 in enumerate(theta0s):
            for j,omega0 in enumerate(omega0s):
                sol_exact = simulate(chain=exact_chain, theta0=theta0,omega0=omega0,t_span=t_span,num_points=num_points)
                thetas_exact[i,j] = sol_exact.y[:num_pends]
                omegas_exact[i,j] = sol_exact.y[num_pends:]
                energy_exact[i,j] = np.array([exact_chain["Exact Solution"].energy(theta,omega) for theta, omega in zip(thetas_exact,omegas_exact)])

                for k,(name,chain) in enumerate(chains.items()):
                     sol = simulate(chain=chain, theta0=theta0,omega0=omega0,t_span=t_span,num_points=num_points)
                     thetas = sol.y[:num_pends]
                     omegas = sol.y[num_pends:]
                     energies = np.array([chain.energy(theta,omega) for theta, omega in zip(thetas,omegas)])
                     theta_diffs[i,j,k] = thetas - thetas_exact[i,j]
                     omegas_diffs[i,j,k] = omegas - omegas_exact[i,j]
                     energy_diffs[i,j,k] = energies - energy_exact[i,j]
    t_vals = np.linspace(t_span[0],t_span[1], num_points)
    return [t_vals,orders,theta_diffs, thetas_exact, omegas_diffs, omegas_exact,energy_diffs,energy_exact,theta0s,omega0s]
                
def plot_theta_diffs(t_vals, diffs, exact_vals, orders, theta0s,omega0s,absolute=True):
    rows = diffs.shape[1]
    cols = diffs.shape[2]
    depth = diffs.shape[0]
    if absolute:
        fig, axs = plt.subplots(rows,cols,figsize = (3*rows,3*cols))   
        for i in range(rows):
             for j in range(cols):
                  axs[i,j].set_title(f"{theta0s[i],omega0s[j]}")
                  for k in range(depth):  
                       axs[i,j].plot(t_vals,diffs[i,j,k],label=f"Taylor Approx N = {orders[k]}")
        axs.set_xlabel("Time(s)")
        axs.set_ylabel("Theta Difference (Rad)")
                     
                 

pendulums = [uniform_massive_rod(m=1,L=1) for _ in range(2)]
theta0s = np.linspace(0,np.pi/2,5)
omega0s = 0.0
diff_vals = angle_energy_diff(pendulums, theta0s, omega0s)

    