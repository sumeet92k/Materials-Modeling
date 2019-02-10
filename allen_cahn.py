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
TUTORIAL 9: Allen-Cahn 
Plots the diffuse interface profile by solving the Allen-Cahn equation. 
Also, adds a biasing term to the free energies to make the interface move (commented).
"""

import numpy as np
import matplotlib.pyplot as plt

W, M, K = 1, 1, 1
mesh_x = 100
dx = 1
dt = 0.005
t_every = 100
timesteps = 1000

s = np.zeros(mesh_x)
s[0:mesh_x//2] = 1 # initialize s 

def free_energy(s):
    return W*s**2*(1-s)**2 #+ s**2*(3-2*s)

def d_free_energy(s):
    return 2*W*s*(s-1)*(2*s-1) #+ 6*s*(1-s)

def calc_laplacian(s):
    laplacian_s = np.zeros(mesh_x)
    for i in range(1, mesh_x-2):
        laplacian_s[i] = (s[i+1] -2*s[i] + s[i-1])/(dx**2)
    
    return laplacian_s

def calc_surface_energy(s):
    ana_SE = np.sqrt(K*W)/3
    grad_s = np.zeros(mesh_x)
    for i in range(1, mesh_x-1):
        grad_s[i] = (s[i+1] - s[i-1])/(2*dx)
    
    grad_SE = 2*K*np.sum(grad_s**2)*dx
    free_energy_SE = 2*np.sum( free_energy(s) )*dx
    
    print(ana_SE, grad_SE, free_energy_SE)

for t in range(timesteps):
    laplacian_s = calc_laplacian(s)
    ds_dt = -M*( d_free_energy(s) - 2*K*laplacian_s )
    s = s + ds_dt*dt
    
    if t%t_every == 0:
        plt.plot(s, '-^', label='t=' + str(t) )

    
calc_surface_energy(s)
plt.legend()    
plt.show()


        