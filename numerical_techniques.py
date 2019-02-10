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
Created on Fri Aug 17 15:07:19 2018
Tutorial 1: numerical techniques
Integrates a function using different techniques: explicit, implicit, predictor-
corrector and 4th-order Range-Kutta
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.01*2
timesteps = int(100/2) # this should always be an integer

t_end = (timesteps-1)*dt # we subtract 1 because t starts from 0

t_array = np.linspace(0, t_end, timesteps) # creates a linearly spaced array, input arguments: np.linspace(start, stop, number of intervals)

# declaring and initializing the arrays for analytical solution
x_ana = np.zeros(timesteps) # initializes an array to 0, input arguments: np.zeros(size of array) 
x_ana = ( 7*np.exp(2*t_array) - 3)*0.5
plt.plot(t_array, x_ana, label='analytical')

# declaring and initializing the arrays
x_exp = np.zeros(timesteps) #explicit
x_exp[0] = 2
x_imp = np.zeros(timesteps) #implicit
x_imp[0] = 2
x_pc = np.zeros(timesteps) #predictor-corrector
x_pc[0] = 2
x_rk4 = np.zeros(timesteps) #Range-Kutta 4th order
x_rk4[0] = 2

for t in range(timesteps-1): # produces sequence of integers from 0 to timesteps-2
    x_exp[t+1] = x_exp[t]*(1 + 2*dt) + 3*dt
    x_imp[t+1] = (x_imp[t] + 3*dt)/(1 - 2*dt)
    
    # predictor-corrector
    x_tilde = x_pc[t]*(1 + 2*dt) + 3*dt # this is not an array
    x_pc[t+1] = x_pc[t] + ( 2*(x_pc[t] + x_tilde)*0.5 + 3 )*dt
    
    # Range-Kutta 4th order
    k1 = dt*( (2*x_rk4[t]) + 3 )
    k2 = dt*( 2*(x_rk4[t] + 0.5*k1) + 3 )
    k3 = dt*( 2*(x_rk4[t] + 0.5*k2) + 3 )
    k4 = dt*( 2*(x_rk4[t] + 1*k3) + 3 )
    x_rk4[t+1] = x_rk4[t] + (k1 + 2*k2 + 2*k3 + k4)/6
    
plt.plot(t_array, x_exp, label='explicit')
plt.plot(t_array, x_imp, label='implicit')
plt.plot(t_array, x_pc, label='predictor-corrector')
plt.plot(t_array, x_rk4, label='Range-Kutta-4')

plt.legend() # necessary for displaying the labels
plt.show() # display the plots in a window