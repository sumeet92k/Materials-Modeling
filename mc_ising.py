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
TUTORIAL 8B: Monte-Carlo Ising model
Explores the magnetization behaviour and the critical magnetization temperature 
using the Monte-Carlo Ising model. Also creates an animation of the spatial 
magnetization state with time as it approaches equilibrium. Finally plots the 
magnetization with time to calculate average magnetization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mesh_x, mesh_y = 50, 50
ims = [] # python array type (different from numpy array type) to save animation images
fig = plt.figure("animation") # create a window for displaying animations

num_ensembles = 200000
arr_magnetization = np.zeros(num_ensembles)

T = 4.0
J = 1.0
B = 0.0

spin = -np.ones((mesh_x, mesh_y))

def calculate_deltaE(x, y):
    spin_left = spin[ (x-1)%mesh_x, y ]
    spin_right = spin[ (x+1)%mesh_x, y ]
    spin_above = spin[ x, (y+1)%mesh_y ]
    spin_below = spin[ x, (y-1)%mesh_y ]
    
    deltaE = 2*J*spin[x, y]*(spin_left + spin_right + spin_above + spin_below) + 2*B*spin[x, y]
    
    return deltaE

for i in range(num_ensembles):
    x, y = np.random.randint(mesh_x), np.random.randint(mesh_y)
    
    deltaE = calculate_deltaE(x, y)
    
    if deltaE < 0.0:
        spin[x, y] = -spin[x, y]
    else:
        rv = np.random.rand()
        if rv < np.exp(-deltaE/T):
            spin[x, y] = -spin[x, y]
    
    avg_magnetization = np.sum(spin)/(mesh_x*mesh_y)
    
    arr_magnetization[i] = avg_magnetization
    
    if i%10000 == 0: # saving animation every 10000 step
        im = plt.imshow(spin, animated=True)
        ims.append([im]) # attaching the animation plot to the array ims

# displaying the animation
ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=1000)
plt.colorbar()


plt.figure("magnetization")
plt.plot(arr_magnetization)
plt.xlabel("steps")
plt.ylabel("magnetization")
plt.grid()

plt.figure("spin state at step =" + str(i))
plt.imshow(spin)
plt.colorbar()
plt.show()