#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2018, sumeet92k
# All rights reserved. Please read the "license.txt" for license terms.
#
# Project Title: Materials Modeling Course Tutorials
#
# Developer: Sumeet Khanna
# Contact Info: https://github.com/sumeet92k

"""
TUTORIAL 4: Diffusion 1D: analytical, explicit, implicit (Thomas algorithm), iterative (Jacobi)
Solves the 1D diffusion equation using analytical, explicit and implicit
(Thomas algorithm) and iterative (Jacobi) techniques. 
The analytical solution is given by the Green's function.
"""

import numpy as np
import matplotlib.pyplot as plt

mesh_x = 60
timesteps = 100

dx = 1.0
dt = 0.1

Diffusivity = 1.0

x_array = np.arange(0, mesh_x, dx)

def comp_analytical(x, t):
    x_prime = mesh_x*dx/2
    c_ana = np.exp( -(x - x_prime)**2/(4*Diffusivity*t*dt) )
    c_ana = c_ana/np.sqrt(4*np.pi*Diffusivity*t*dt)
    
    return c_ana

def initialize(c):
    c[mesh_x//2] = 1.0
    
def apply_bc(c):
    c[0] = 0.0
    c[mesh_x-1] = 0.0
    
# explicit solution
c_exp, c_exp_new = np.zeros(mesh_x), np.zeros(mesh_x)
initialize(c_exp)
apply_bc(c_exp)

for t in range(timesteps):
    for i in range(1, mesh_x-1):
        dcdt = Diffusivity*(c_exp[i-1] - 2*c_exp[i] + c_exp[i+1])/(dx**2)
        c_exp_new[i] = c_exp[i] + dcdt*dt
    c_exp = c_exp_new
    apply_bc(c_exp)    
    
# implicit solultion: Thomas algorithm
c_imp, c_imp_new = np.zeros(mesh_x), np.zeros(mesh_x)
A, B, C, D = np.zeros(mesh_x), np.zeros(mesh_x), np.zeros(mesh_x), np.zeros(mesh_x)
alpha = Diffusivity*dt/(dx**2)

initialize(c_imp)
for t in range(timesteps):
    apply_bc(c_imp)
    A[1:mesh_x-1] = -alpha
    B[1:mesh_x-1] = 1 + 2*alpha
    C[1:mesh_x-1] = -alpha
    D[1:mesh_x-1] = c_imp[1:mesh_x-1]
    
    D[1] = D[1] - A[1]*c_imp[0]
    A[1] = 0
    
    D[mesh_x-2] = D[mesh_x-2] - C[mesh_x-2]*c_imp[mesh_x-1]
    C[mesh_x-2] = 0
    
    # forward transform of coefficients
    for i in range(2, mesh_x-1):
        factor = A[i]/B[i-1]
        B[i] = B[i] - factor*C[i-1]
        D[i] = D[i] - factor*D[i-1]
        A[i] = 0
        
    # backward elimination
    c_imp[mesh_x-2] = D[mesh_x-2]/B[mesh_x-2]
    for i in range(mesh_x-3, 0, -1):
        c_imp[i] = (D[i] - C[i]*c_imp[i+1])/B[i]

# jacobi solution
c_jac, c_jac_new = np.zeros(mesh_x), np.zeros(mesh_x)
initialize(c_jac)
numiters = 10
for t in range(timesteps):
    apply_bc(c_jac)
    A[1:mesh_x-1] = -alpha
    B[1:mesh_x-1] = 1 + 2*alpha
    C[1:mesh_x-1] = -alpha
    D[1:mesh_x-1] = c_jac[1:mesh_x-1]
    
    for iterations in range(numiters):
        for i in range(1, mesh_x-1):
            c_jac_new[i] = (D[i] - A[i]*c_jac[i-1] - C[i]*c_jac[i+1])/B[i]
        c_jac = c_jac_new
        apply_bc(c_jac)
    

# plotting solutions
c_ana = comp_analytical(x_array, timesteps-1)
plt.plot(x_array, c_ana, label = 'ana')
plt.plot(x_array, c_exp, label = 'exp')
plt.plot(x_array, c_imp, label = 'imp')
plt.plot(x_array, c_jac, '-*', label = 'jac')


plt.legend()    
plt.show()    