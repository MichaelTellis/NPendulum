#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:53 2026

@author: adlerbliske
"""

#import neccesary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

#define variable parameters
m1 = m2 = 1
r1 = r2 = 1
runtime = 75
framerate = 60
th1_inital = th2_initial = 1
dot_th1_initial=0
dot_th2_initial=0
g = 9.80665 #45 latitude sea level

#scpecify time frame for integration
initial_state = [th1_inital, dot_th1_initial, th2_initial, dot_th2_initial]
t_eval = np.linspace(0, runtime, runtime*framerate)

#create a function for scipy to integrate
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

#solve ivp for real
L1 = solve_ivp(double_pendulum, (0, runtime), initial_state, t_eval=t_eval)
th1 = L1.y[0]
dot_th1 = L1.y[1]
th2 = L1.y[2]
dot_th2 = L1.y[3]

x1 = r1 * np.sin(th1)
y1 = -1 * r1 * np.cos(th1)
x2 = r1 * np.sin(th1) + r2 * np.sin(th2)
y2 = -1 * r1 * np.cos(th1) - r2 * np.cos(th2)

#solve ivp for SAE
L2 = solve_ivp(double_pendulum_SAE, (0, runtime), initial_state, t_eval=t_eval)
th1_SAE = L2.y[0]
dot_th1_SAE = L2.y[1]
th2_SAE = L2.y[2]
dot_th2_SAE = L2.y[3]

        
x1_SAE = r1 * np.sin(th1_SAE)
y1_SAE = -1 * r1 * np.cos(th1_SAE)
x2_SAE = r1 * np.sin(th1_SAE) + r2 * np.sin(th2_SAE)
y2_SAE = -1 * r1 * np.cos(th1_SAE) - r2 * np.cos(th2_SAE)

#remap y values to origin at (0, r1+r2)
y1+= r1 + r2
y2 += r1 + r2
y1_SAE += r1 + r2
y2_SAE += r1 + r2

#find ds
ds_array = np.sqrt((x2_SAE - x2)**2 + (y2_SAE - y2)**2)

#find values for dp windows
x_min = np.min([x2, x2_SAE])
x_max = np.max([x2, x2_SAE])
y_min = np.min([y2, y2_SAE])
y_max = np.max([y2, y2_SAE])
ds_max = np.max(ds_array)

#define mosaic
plot, ax = plt.subplot_mosaic([['A', 'C'], ['A', 'B']], layout = 'constrained')

#define plots
dp, = ax['A'].plot(0, 0, marker='o', markerfacecolor='r', color='k')
dp_SAE, = ax['A'].plot(0, 0, marker='o', markerfacecolor='b', color='k')
path, = ax['A'].plot(0,0, color='r', alpha=0.2)
path_SAE, = ax['A'].plot(0,0, color='b', alpha=0.2)

phase_space2, = ax['B'].plot(0,0, color='r', alpha=0.5)
phase_space1, = ax['C'].plot(0,0, color='r', alpha=0.5)

phase_space2_SAE, = ax['B'].plot(0,0, color='b', alpha=0.5)
phase_space1_SAE, = ax['C'].plot(0,0, color='b', alpha=0.5)

#set window
ax['A'].set_aspect('equal')
ax['A'].set_xlim(x_min - 0.1, x_max + 0.1)
if y_max < r1 + r2: ax['A'].set_ylim(y_min - 0.1 , (r1 + r2) + 0.1)
else: ax['A'].set_ylim(y_min - 0.1, y_max + 0.1)

ax['B'].set_ylim(np.min([dot_th2, dot_th2_SAE]), 
                 np.max([dot_th2, dot_th2_SAE]))
ax['B'].set_xlim(np.min([th2, th2_SAE]), 
                 np.max([th2, th2_SAE]))

ax['C'].set_ylim(np.min([dot_th1, dot_th1_SAE]), 
                 np.max([dot_th1, dot_th1_SAE]))
ax['C'].set_xlim(np.min([th1, th1_SAE]), 
                 np.max([th1, th1_SAE]))

#animate
def animate(i):
    dp.set_xdata([0, x1[i], x2[i]])
    dp.set_ydata([r1+r2, y1[i], y2[i]])
    path.set_xdata(x2[:i])
    path.set_ydata(y2[:i])

    dp_SAE.set_xdata([0, x1_SAE[i], x2_SAE[i]])
    dp_SAE.set_ydata([r1+r2, y1_SAE[i], y2_SAE[i]])
    path_SAE.set_xdata(x2_SAE[:i])
    path_SAE.set_ydata(y2_SAE[:i])
    
    phase_space2_SAE.set_ydata(dot_th2_SAE[:i])
    phase_space2_SAE.set_xdata(th2_SAE[:i])
    
    phase_space2.set_ydata(dot_th2[:i])
    phase_space2.set_xdata(th2[:i])
    
    phase_space1_SAE.set_ydata(dot_th1_SAE[:i])
    phase_space1_SAE.set_xdata(th1_SAE[:i])
    
    phase_space1.set_ydata(dot_th1[:i])
    phase_space1.set_xdata(th1[:i])


    return dp, path, dp_SAE, path_SAE, phase_space1, phase_space1_SAE, phase_space2, phase_space2_SAE

anim = FuncAnimation(plot, animate, frames=runtime*framerate, interval=1000/framerate, blit=True)
plt.show()