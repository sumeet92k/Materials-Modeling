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
TUTORIAL 7: Diffusion Limited Aggregation
Simulates Dendritic single crystal growth from an initial seed at the centre using 
Diffusion Limited Aggregation (DLA).
"""

import numpy as np
import matplotlib.pyplot as plt


def initialize_particle(r_start):
    theta = 2*np.pi*np.random.random()
    x = x_centre + r_start*np.cos(theta)
    y = y_centre + r_start*np.sin(theta)
    
    x, y = int(x), int(y)
    return x, y

def jump(x, y):
    random_number = np.random.random()
    
    if random_number < 0.25:
        # jump left
        x = x - 1
        y = y
    elif random_number < 0.5:
        x = x + 1
        y = y
        
    elif random_number < 0.75:
        # jump above
        x = x
        y = y + 1
    else:
        # jump below
        x = x
        y = y - 1
        
    return x, y

def check_neighbours(x, y):
    if phase[x+1, y] == 1 or phase[x-1, y] == 1 or phase[x, y-1] == 1 or phase[x, y+1] == 1:
        return "solid"
    else:
        return "liquid"
    
def check_boundary(x, y):
    if 0 < x < mesh_x - 1 and 0 < y < mesh_y - 1:
        return "inside"
    else:
        return "outside"
    

mesh_x, mesh_y = 100, 100
phase = np.zeros((mesh_x, mesh_y))
x_centre, y_centre = mesh_x//2, mesh_y//2

phase[x_centre, y_centre] = 1

r_max = 0
r_start = r_max + 5

while r_start < mesh_x//2 - 1:
    x, y = initialize_particle(r_start)
    
    while check_neighbours(x, y) == "liquid":
        x, y = jump(x, y)
        if check_boundary(x, y) == "outside":
            break
        if check_neighbours(x, y) == "solid":
            phase[x, y] = 1
            if (x-x_centre)**2 + (y-y_centre)**2 > r_max**2:
                r_max = np.sqrt( (x-x_centre)**2 + (y-y_centre)**2 )
                r_start = r_max + 5

plt.imshow(phase)
plt.colorbar()
plt.show()