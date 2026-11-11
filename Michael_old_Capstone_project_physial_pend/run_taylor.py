import numpy as np
import matplotlib.pyplot as plt
from NPhysicsPendulumSim import pendulumChain, uniform_massive_rod, simulate, plotting_routine, animate
from TaylorPhysicalPendulumSim import TaylorChain

'''
Runs a plotting routine for each pendulum, you can customize it.
Defauult I have  2 uniform massive rods linked together.
Default plots exact sol, N = 1 approx, and N=3 approx
No comparison modules yet, and I also want to do stuff looking at energy. 

'''

#Setup initial conditions

pendulum = uniform_massive_rod(m=1,L=1)
pendulums = [pendulum,pendulum]
theta0 = np.pi/4
omega0 = 0.0
t_span = (0,10)
num_points = 2000
g = 9.81

#build the three models:
chains = {
    "Exact Solution": pendulumChain(pendulums,g=g),
    "Taylor Approx N = 1": TaylorChain(pendulums,order=1,g=g),
    "Taylor Approx N = 3": TaylorChain(pendulums,order=3,g=g)
}
animations = [None] * len(chains)
t_vals = np.linspace(t_span[0],t_span[1],num_points)
#run simulation and plot them seperately.
#I'd like to make a function that allows you to compare them
#Also a function to show error over time of different taylor approx, starting conditions, and physical models.
for i, (name,chain) in zip(range(len(chains)),chains.items()):
    
    sol = simulate(chain=chain, theta0=theta0,omega0=omega0,t_span=t_span,num_points=num_points)
    thetas = sol.y[:len(pendulums)]
    omegas = sol.y[len(pendulums):]
    thetas = thetas.T
    omegas = omegas.T
    energy = [chain.energy(theta,omega) for theta, omega in zip(thetas,omegas)]
    plt.plot(t_vals, energy, label= f"{name}")
    
    print(f"{name}: {sol.message}")
    #plotting_routine(chain,sol,save_as=f"{name.replace(' ','_')}.png")

    #animations[i] = animate(chain, sol, step=2, save_as=f"{name.replace(' ', '_')}.mp4",title=name)  


#print(energies[0])
#plt.plot(t_vals, energies[2])

plt.legend()



plt.show()