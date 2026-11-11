#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:33:30 2026

@author: adlerbliske
"""

#import neccesary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

#define variable parameters
m1_range=(1,1)
m2_range=(1,1)
r1_range=(1,1)
r2_range=(1,1)
t_range=(0,10)
th1_inital=2
th2_initial=2
dot_th1_initial=0
dot_th2_initial=0
iterations=1

g = 9.81
M1 = list(np.linspace(m1_range[0], m1_range[1], iterations))
M2 = list(np.linspace(m2_range[0], m2_range[1], iterations))
R1 = list(np.linspace(r1_range[0], r1_range[1], iterations))
R2 = list(np.linspace(r2_range[0], r2_range[1], iterations))

#scpecify time frame for integration
initial_state = [th1_inital, dot_th1_initial, 
                     th2_initial, dot_th2_initial]
t_eval = np.linspace(t_range[0], t_range[1], t_range[1]*60)

#create a function for scipy to integrate
def double_pendulum(t, state):
        th1, dot_th1, th2, dot_th2 = state

#construct matrices
        M = ([(m1 + m2) * r1, m2 * r2 * np.cos(th1 - th2)],
             [r1 * np.cos(th1 - th2), r2])
            
        K = ([-1 * m2 * r2 * (dot_th2 ** 2) * np.sin(th1 - th2) - (m1 + m2) * g * np.sin(th1),
              r1 * (dot_th1 ** 2) * np.sin(th1 - th2) - g * np.sin(th2)])

#let numpy deconstruct the coupled equations
        ddot_th1, ddot_th2 = np.linalg.solve(M, K)
        return [dot_th1, ddot_th1, dot_th2, ddot_th2]

for i in range(iterations):
        m1 = M1[i]
        m2 = M2[i]
        r1 = R1[i]
        r2 = R2[i]
        solution = solve_ivp(double_pendulum, 
                             t_range, 
                             initial_state, 
                             t_eval=t_eval)
        th1 = solution.y[0]
        th2 = solution.y[2]
        
        x1 = r1 * np.sin(th1)
        y1 = -1 * r1 * np.cos(th1)
        x2 = r1 * np.sin(th1) + r2 * np.sin(th2)
        y2 = -1 * r1 * np.cos(th1) - r2 * np.cos(th2)

canvas, ax = plt.subplots()
dp, = ax.plot(0, 0, marker='o', markerfacecolor='r', color='k')
path, = ax.plot(0,0, color='red', alpha=0.2)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-0.2, 3.2)

def func(i):
                dp.set_xdata([0, x1[i], x2[i]])
                dp.set_ydata([2, y1[i]+2, y2[i]+2])
                path.set_xdata(x2[:i])
                path.set_ydata(y2[:i]+2)
                return dp, path,

anim = FuncAnimation(canvas, func, frames=t_range[1]*60, interval=1000/60, blit=True)
plt.show()