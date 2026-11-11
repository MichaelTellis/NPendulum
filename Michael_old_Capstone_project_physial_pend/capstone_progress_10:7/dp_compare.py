#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:09:30 2026

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
time = 75
t_range = (0,time)
th1_inital = th2_initial = 0.75
#third term of taylor exansion becomes noticable around 0.2 radians
#collapses quickly at 1 radian
dot_th1_initial=0
dot_th2_initial=0
g = 9.80665 #45 latitude sea level

#scpecify time frame for integration
initial_state = [th1_inital, dot_th1_initial, th2_initial, dot_th2_initial]
t_eval = np.linspace(t_range[0], t_range[1], t_range[1]*60)

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
L1 = solve_ivp(double_pendulum, t_range, initial_state, t_eval=t_eval)
th1 = L1.y[0]
th2 = L1.y[2]

x1 = r1 * np.sin(th1)
y1 = -1 * r1 * np.cos(th1)
x2 = r1 * np.sin(th1) + r2 * np.sin(th2)
y2 = -1 * r1 * np.cos(th1) - r2 * np.cos(th2)

#solve ivp for SAE
L2 = solve_ivp(double_pendulum_SAE, t_range, initial_state, t_eval=t_eval)
th1_SAE = L2.y[0]
th2_SAE = L2.y[2]
        
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
plot, ax = plt.subplot_mosaic([['B', 'B', 'B'],['A', 'C', 'C']], layout = 'constrained')

#define plots
dp, = ax['A'].plot(0, 0, marker='o', markerfacecolor='r', color='k')
dp_SAE, = ax['A'].plot(0, 0, marker='o', markerfacecolor='b', color='k')
path, = ax['A'].plot(0,0, color='r', alpha=0.2)
path_SAE, = ax['A'].plot(0,0, color='b', alpha=0.2)
ds_line, = ax['A'].plot(0,0,color='purple', alpha=0.7)

x2_v_t, = ax['B'].plot(0,0, color='r', alpha=0.75)
x2_SAE_v_t, = ax['B'].plot(0,0, color='b', alpha=0.5)

ds, = ax['C'].plot(0,0,color='purple', alpha=0.7)

#set window
ax['A'].set_aspect('equal')
ax['A'].set_xlim(x_min - 0.1, x_max + 0.1)
if y_max < r1 + r2: ax['A'].set_ylim(y_min - 0.1 , (r1 + r2) + 0.1)
else: ax['A'].set_ylim(y_min - 0.1, y_max + 0.1)

ax['B'].set_xlim(0,time)
ax['B'].set_ylim(x_min, x_max)

ax['C'].set_xlim(0,time)
ax['C'].set_ylim(0,ds_max)

#animate
def animate(i):
    dp.set_xdata([0, x1[i], x2[i]])
    dp.set_ydata([r1+r2, y1[i], y2[i]])
    #path.set_xdata(x2[:i])
    #path.set_ydata(y2[:i])
    

    dp_SAE.set_xdata([0, x1_SAE[i], x2_SAE[i]])
    dp_SAE.set_ydata([r1+r2, y1_SAE[i], y2_SAE[i]])
    #path_SAE.set_xdata(x2_SAE[:i])
    #path_SAE.set_ydata(y2_SAE[:i])
  
    ds_line.set_xdata([x2[i], x2_SAE[i]])
    ds_line.set_ydata([y2[i], y2_SAE[i]])

    x2_v_t.set_ydata(x2[:i])
    x2_SAE_v_t.set_ydata(x2_SAE[:i])
    x2_v_t.set_xdata(L1.t[:i])
    x2_SAE_v_t.set_xdata(L1.t[:i])
    
    ds.set_xdata(L1.t[:i])
    ds.set_ydata(ds_array[:i])

    return dp, path, dp_SAE, path_SAE, x2_v_t, x2_SAE_v_t, ds, ds_line

anim = FuncAnimation(plot, animate, frames=t_range[1]*60, interval=1000/60, blit=True)
plt.show()