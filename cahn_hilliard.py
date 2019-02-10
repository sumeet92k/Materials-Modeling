#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2019, sumeet92k
# All rights reserved. Please read the "license.txt" for license terms.
#
# Project Title: Materials Modeling Course Tutorials
#
# Developer: Sumeet Khanna
# Contact Info: https://github.com/sumeet92k

"""
TUTORIAL 10: Cahn-Hilliard Equation 
Solves the Cahn-Hilliard equation for different cases of initialization of the 
initial composition (c). Plots the composition profile with time.
"""

import numpy as np
import matplotlib.pyplot as plt

W, M, K = 1, 1, 1
mesh_x = 200
dx = 1
dt = 0.005
t_every = 20000
timesteps = 200000

def free_energy(c):
    return W*c**2*(1-c)**2 #+ c**2*(3-2*c)

def d_free_energy(c):
    return 2*W*c*(c-1)*(2*c-1) #+ 6*c*(1-c)

def calc_laplacian(c):
    laplacian_c = np.zeros(mesh_x)
    for i in range(1, mesh_x-2):
        laplacian_c[i] = (c[i+1] -2*c[i] + c[i-1])/(dx**2)
    
    return laplacian_c

def calc_surface_energy(c):
    ana_SE = np.sqrt(K*W)/3
    grad_c = np.zeros(mesh_x)
    for i in range(1, mesh_x-1):
        grad_c[i] = (c[i+1] - c[i-1])/(2*dx)
    
    grad_SE = 2*K*np.sum(grad_c**2)*dx
    free_energy_SE = 2*np.sum( free_energy(c) )*dx
    
    print(t, ana_SE, grad_SE, free_energy_SE)

def impose_PBC(c):
    c[0] = c[mesh_x-4]
    c[1] = c[mesh_x-3]
    
    c[mesh_x-2] = c[2]
    c[mesh_x-1] = c[3]
    
    return c

def initialize(c):
    amplitude = 0.1
    wavelength = mesh_x/6
    
    for i in range(mesh_x):
        c[i] = 0.5 + amplitude*np.sin(np.pi*i/wavelength)
    return c

def initialize_random(c):
    amplitude = 0.1
    for i in range(mesh_x):
        c[i] = 0.5 + amplitude*np.random.random()
    return c

c = np.zeros(mesh_x)
#c[0:mesh_x//2] = 1 # initialize c
# c = initialize(c)
c = initialize_random(c)
 
c = impose_PBC(c)

for t in range(timesteps):
    laplacian_c = calc_laplacian(c)
    mu = d_free_energy(c) - 2*K*laplacian_c
    laplacian_mu = calc_laplacian(mu)
    
    dc_dt = M*laplacian_mu
    c = c + dc_dt*dt
    c = impose_PBC(c)
    
    if t%t_every == 0:
        plt.plot(c, '-^', label='t=' + str(t) )
        calc_surface_energy(c)


    
plt.legend()    
plt.show()
