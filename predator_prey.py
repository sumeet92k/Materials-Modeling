#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018, sumeet92k
# All rights reserved. Please read the "license.txt" for license terms.
#
# Project Title: Materials Modeling Course Tutorials
#
# Developer: Sumeet Khanna
# Contact Info: https://github.com/sumeet92k
#
"""
TUTORIAL 2: predator prey model
Solves the coupled Lodka-Volterra (predator-prey) equations using different
numerical techniques
"""

import numpy as np
import matplotlib.pyplot as plt

alpha = 4
beta = 2
delta = 3
gamma = 3

dt = 0.001
timesteps = 20000

t_end = dt*(timesteps -1)
t_array = np.linspace(0, t_end, timesteps)


def get_dx_dt(x, y):
    return alpha*x - beta*x*y

def get_dy_dt(x, y):
    return delta*x*y - gamma*y


x_exp = np.zeros(timesteps)
y_exp = np.zeros(timesteps)

x_pc = np.zeros(timesteps)
y_pc = np.zeros(timesteps)

x_rk4 = np.zeros(timesteps)
y_rk4 = np.zeros(timesteps)

x_exp[0] = 10
y_exp[0] = 5

x_pc[0] = 10
y_pc[0] = 5

x_rk4[0] = 10
y_rk4[0] = 5

for t in range(timesteps - 1):
    # explicit method
    x_exp[t+1] = x_exp[t] + get_dx_dt(x_exp[t], y_exp[t] )*dt
    y_exp[t+1] = y_exp[t] + get_dy_dt(x_exp[t], y_exp[t] )*dt
    
    # predictor corrector method
    x_tilde = x_pc[t] + get_dx_dt(x_pc[t], y_pc[t])*dt
    y_tilde = y_pc[t] + get_dy_dt(x_pc[t], y_pc[t])*dt
    
    x_pc[t+1] = x_pc[t] + 0.5*( get_dx_dt(x_pc[t], y_pc[t]) + get_dx_dt(x_tilde, y_tilde) )*dt
    y_pc[t+1] = y_pc[t] + 0.5*( get_dy_dt(x_pc[t], y_pc[t]) + get_dy_dt(x_tilde, y_tilde) )*dt
    
    # RK 4 method
    k1_x = dt*get_dx_dt(x_rk4[t], y_rk4[t])
    k1_y = dt*get_dy_dt(x_rk4[t], y_rk4[t])
    
    x_tilde = x_rk4[t] + 0.5*k1_x
    y_tilde = y_rk4[t] + 0.5*k1_y
    
    k2_x = dt*get_dx_dt(x_tilde, y_tilde)
    k2_y = dt*get_dy_dt(x_tilde, y_tilde)
    
    x_tilde = x_rk4[t] + 0.5*k2_x
    y_tilde = y_rk4[t] + 0.5*k2_y
    
    k3_x = dt*get_dx_dt(x_tilde, y_tilde)
    k3_y = dt*get_dy_dt(x_tilde, y_tilde)
    
    x_tilde = x_rk4[t] + 1*k3_x
    y_tilde = y_rk4[t] + 1*k3_y
    
    k4_x = dt*get_dx_dt(x_tilde, y_tilde)
    k4_y = dt*get_dy_dt(x_tilde, y_tilde)
    
    x_rk4[t+1] = x_rk4[t] + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
    y_rk4[t+1] = y_rk4[t] + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
    
plt.plot(x_exp, y_exp, label='explicit')
plt.plot(x_pc, y_pc, label='predictor-corrector')
plt.plot(x_rk4, y_rk4, label='RK4')

plt.xlabel("deer") # label the axes
plt.ylabel("lion") 
plt.legend()
plt.show()

# plotting lion and deer population with time in a different figure
plt.figure()
plt.plot(t_array, x_rk4, label='deer')
plt.plot(t_array, y_rk4, label='lion')
plt.xlabel("time")
plt.ylabel("population")
plt.legend()
plt.show()