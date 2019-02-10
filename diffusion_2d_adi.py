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
TUTORIAL 5: Diffusion 2D: ADI (Alternate Direction Implicit) scheme
Solves the 2D diffusion equation using Alternate Direction Implicit scheme and 
plots the composition profiles in 2D and 3D plots
"""

import numpy as np
import matplotlib.pyplot as plt

Diffusivity = 1.0
timesteps = 100
dt = 0.1
dx = 1.0

mesh_x, mesh_y = 50, 50

def initialize(c):
    c[mesh_x//2, mesh_y//2] = 1.0

def apply_bc(c):
    c[0, :] = 0.0
    c[mesh_x-1, :] = 0.0
    c[:, 0] = 0.0
    c[:, mesh_y-1] = 0.0


# implicit solultion: Thomas algorithm
c_imp = np.zeros( (mesh_x, mesh_y) )

A_x, B_x, C_x, D_x = np.zeros(mesh_x), np.zeros(mesh_x), np.zeros(mesh_x), np.zeros(mesh_x)
A_y, B_y, C_y, D_y = np.zeros(mesh_y), np.zeros(mesh_y), np.zeros(mesh_y), np.zeros(mesh_y)
alpha = Diffusivity*dt/(dx**2)

initialize(c_imp)
for t in range(timesteps):
    # implicit in x, explicit in y
    apply_bc(c_imp)
    for jj in range(1, mesh_y-1):
        
        A_x[1:mesh_x-1] = -alpha
        B_x[1:mesh_x-1] = 1 + 4*alpha
        C_x[1:mesh_x-1] = -alpha
        D_x[1:mesh_x-1] = c_imp[1:mesh_x-1, jj] + alpha*c_imp[1:mesh_x-1, jj-1] + alpha*c_imp[1:mesh_x-1, jj+1]
        
        D_x[1] = D_x[1] - A_x[1]*c_imp[0, jj]
        A_x[1] = 0
        
        D_x[mesh_x-2] = D_x[mesh_x-2] - C_x[mesh_x-2]*c_imp[mesh_x-1, jj]
        C_x[mesh_x-2] = 0
        
        # forward transform of coefficients
        for i in range(2, mesh_x-1):
            factor = A_x[i]/B_x[i-1]
            B_x[i] = B_x[i] - factor*C_x[i-1]
            D_x[i] = D_x[i] - factor*D_x[i-1]
            A_x[i] = 0
            
        # backward elimination
        c_imp[mesh_x-2, jj] = D_x[mesh_x-2]/B_x[mesh_x-2]
        for i in range(mesh_x-3, 0, -1):
            c_imp[i, jj] = (D_x[i] - C_x[i]*c_imp[i+1, jj])/B_x[i]
            
            
    # implicit in y, explicit in x
    apply_bc(c_imp)
    mesh_x, mesh_y = mesh_y, mesh_x
    c_imp = np.transpose(c_imp)
    
    for jj in range(1, mesh_y-1):
        
        A_y[1:mesh_x-1] = -alpha
        B_y[1:mesh_x-1] = 1 + 4*alpha
        C_y[1:mesh_x-1] = -alpha
        D_y[1:mesh_x-1] = c_imp[1:mesh_x-1, jj] + alpha*c_imp[1:mesh_x-1, jj-1] + alpha*c_imp[1:mesh_x-1, jj+1]
        
        D_y[1] = D_y[1] - A_y[1]*c_imp[0, jj]
        A_y[1] = 0
        
        D_y[mesh_x-2] = D_y[mesh_x-2] - C_y[mesh_x-2]*c_imp[mesh_x-1, jj]
        C_y[mesh_x-2] = 0
        
        # forward transform of coefficients
        for i in range(2, mesh_x-1):
            factor = A_y[i]/B_y[i-1]
            B_y[i] = B_y[i] - factor*C_y[i-1]
            D_y[i] = D_y[i] - factor*D_y[i-1]
            A_y[i] = 0
            
        # backward elimination
        c_imp[mesh_x-2, jj] = D_y[mesh_x-2]/B_y[mesh_x-2]
        for i in range(mesh_x-3, 0, -1):
            c_imp[i, jj] = (D_y[i] - C_y[i]*c_imp[i+1, jj])/B_y[i]
    
    mesh_x, mesh_y = mesh_y, mesh_x
    c_imp = np.transpose(c_imp)

# for 3d plotting
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 2D map plot
plt.figure(1) # opens in  a different window
plt.imshow(c_imp, cmap=cm.coolwarm) # cmap is the coloring scheme
plt.colorbar() # adds color legend
    
# 3D surface plot    
fig = plt.figure(2) # opens in  a different window
ax = fig.gca(projection='3d') # adds a third axis
x_array = np.arange(0, mesh_x, dx)
y_array = np.arange(0, mesh_y, dx)
X, Y = np.meshgrid(y_array, x_array) # creates an xy grid of size mesh_x*mesh_y
surf = ax.plot_surface(X, Y, c_imp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()