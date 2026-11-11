#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:25:50 2026

@author: adlerbliske
"""

#import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#define variable parameters
m1 = m2 = 1
r1 = r2 = 1
runtime = 100
framerate = 60
th1_inital = th2_initial = 0.5
dot_th1_initial=0
dot_th2_initial=0
g = 9.80665 #45 latitude @ sea level

#define function to integrate
def double_pendulum(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = ([(m1 + m2) * r1, m2 * r2 * np.cos(th1 - th2)],
         [r1 * np.cos(th1 - th2), r2])
            
    K = ([-1 * m2 * r2 * (dot_th2 ** 2) * np.sin(th1 - th2) - (m1 + m2) * g * np.sin(th1),
         r1 * (dot_th1 ** 2) * np.sin(th1 - th2) - g * np.sin(th2)])

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]

def double_pendulum_SAE(t, state):
    th1, dot_th1, th2, dot_th2 = state

    M = ([(m1 + m2) * r1, m2 * r2],
         [r1, r2])
             
    K = ([-1 * m2 * r2 * (dot_th2 ** 2) * (th1 - th2) - (m1 + m2) * g * th1,
         r1 * (dot_th1 ** 2) * (th1 - th2) - g * th2])

    ddot_th1, ddot_th2 = np.linalg.solve(M, K)
    return [dot_th1, ddot_th1, dot_th2, ddot_th2]


#integrate
initial_state = [th1_inital, dot_th1_initial, th2_initial, dot_th2_initial]
t_eval = np.linspace(0, runtime, runtime*framerate)

solution = solve_ivp(double_pendulum, (0, runtime), initial_state, t_eval=t_eval)
solution_SAE = solve_ivp(double_pendulum_SAE, (0, runtime), initial_state, t_eval=t_eval)

#extract values
th1 = solution.y[0]
dot_th1 = solution.y[1]
th2 = solution.y[2]
dot_th2 = solution.y[3]

th1_SAE = solution_SAE.y[0]
dot_th1_SAE = solution_SAE.y[1]
th2_SAE = solution_SAE.y[2]
dot_th2_SAE = solution_SAE.y[3]
time = solution.t

#calculate lagrangian
#additional negative sign accounts for pendulums hanging from the origin
L = 0.5 * (m1+m2) * r1**2 * dot_th1**2\
+ 0.5 * m2 * r2**2 * dot_th2**2\
+ m2 * r1 * r2 * dot_th1 * dot_th2 * np.cos(th1 - th2)\
- (m1 + m2) * g * r1 * np.cos(th1)\
- m2 * g * r2 * np.cos(th2)

L_SAE = 0.5 * (m1+m2) * r1**2 * dot_th1_SAE**2\
+ 0.5 * m2 * r2**2 * dot_th2_SAE**2\
+ m2 * r1 * r2 * dot_th1_SAE * dot_th2 * np.cos(th1_SAE - th2_SAE)\
- (m1 + m2) * g * r1 * np.cos(th1_SAE)\
- m2 * g * r2 * np.cos(th2_SAE)

#additional negative sign accounts for pendulums hanging from the origin
L = -L
L_SAE = -L_SAE

#plot
plt.plot(time, L, color='red')
plt.plot(time, L_SAE, color='blue', alpha=0.5)
m, b = np.polyfit(time, L_SAE, 1)
line_of_best_fit = m * time + b #y=mx+b
plt.plot(time, line_of_best_fit, color='blue')
