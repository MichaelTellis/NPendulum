import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from NPhysicsPendulumSim import pendulumChain, Pendulum, uniform_massive_rod
from math import factorial


'''
New math, gonna be honest, not sure if its correct
From exact eqs in NPhysicsPendulumSim.py:
Equations of motion:

sum_j M_ij theta_j'' = - sum_j A_ij sin(theta_i - theta_j) omega_j^2 - g B_i sin(theta_i)

where:

M_ij = A_ij cos(theta_i - theta_j) + kronecker_delta_ij * I_cm_i


'''
def sin_taylor(x,N):
    '''
    sinx degree N. Returns value of taylor series for each value of x by adding terms to the n power
    x: numpy array
    N: int
    '''

    total = np.zeros_like(x,dtype=float)
    for i in range(1,N+1,2):
        total  += (-1)**((i-1)/2)*(x**i)/factorial(i)
    return total


def cos_taylor(x,N):
    '''
    cosx degree N. Returns values of each taylor series as an array
    '''
    total = np.zeros_like(x,dtype=float)
    for i in range(0,N+1, 2):
        total += (-1)**(i/2)*(x**i)/factorial(i)
    return total

class TaylorChain(pendulumChain):
    def __init__(self, pendulums: list,order: int, g=9.81):
        '''
        Takes in the same parameter data as the PendulumChain Object so I'm making it a subclass.

        The things that need to change are all of the sines and cosines.
        order: int>=1
            Idk what happens if you put a negative num, it probably explodes
        
        '''
        super().__init__(pendulums, g)
        self.order = order

    def state(self,t,y):
        '''
        solves rhs for solve_ivp but using N-order approximations now.
         y = [theta_0..theta_{n-1}, omega_0..omega_{n-1}].

         Solves the new equation of motion for theta: 

         sum_j M_ij theta_j'' = - sum_j A_ij taylor_sin(theta_i - theta_j) omega_j^2 - g B_i taylor_sin(theta_i)

        where:

        M_ij = A_ij taylor_cos(theta_i - theta_j) + kronecker_delta_ij * I_cm_i
        '''
         # extract out prev theta and omega states
        theta = y[:self.n]
        omega = y[self.n:]
        # pairwise differences of thetas
        diff =np.subtract.outer(theta, theta)
        #Calculate M matrix 
        M = self.A * cos_taylor(diff,self.order) + self.I_diag
        rhs = -((self.A*sin_taylor(diff,self.order)) @ (omega * omega)) - self.g_b * sin_taylor(theta,self.order)

        result = np.empty(2*self.n)
        # d theta /dt = omega
        result[:self.n] = omega
        # Solve the matrix equation to find theta'', aka d omega /dt.
        #Solve Ax = b, where A = M and b = rhs, x = theta''.
        result[self.n:] = np.linalg.solve(M,rhs)
        return result

    def taylor_energy(self,theta,omega):
        '''
        calculates energy of the system at a given time.
        theta, omega: arrays of shape (n,)
        E_N = K + U
        E_N = (1/2 omega^T * M * omega ) + (g * sum_i B_i * (1 - cos_taylor(theta_i)))

        M_ij = A_ij taylor_cos(theta_i - theta_j) + kronecker_delta_ij * I_cm_i
        
        '''
        diff = np.subtract.outer(theta,theta)
        M = self.A * cos_taylor(diff, self.order) + self.I_diag
        KE = 0.5 * (omega @ M @ omega)
        PE = self.g_b * (1- cos_taylor(theta,self.order))
        return KE+PE

    




