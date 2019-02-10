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
TUTORIAL 6: Monte Carlo Random-Walker 
Finds the relationship between mean squared displacement and time for a random 
walker using Monte-Carlo technique
"""


import numpy as np
import matplotlib.pyplot as plt
import random

nexperiments = 1000
njumps = 200
x, y = np.zeros(njumps), np.zeros(njumps)

def perform_exp():
    x[0], y[0] = 0, 0
    
    for i in range(1, njumps):
        random_number = np.random.random()
        
        if random_number < 0.25:
            # jump left
            x[i] = x[i-1] - 1
            y[i] = y[i-1]
        elif random_number < 0.5:
            x[i] = x[i-1] + 1
            y[i] = y[i-1]
            
        elif random_number < 0.75:
            # jump above
            x[i] = x[i-1]
            y[i] = y[i-1] + 1
        else:
            # jump below
            x[i] = x[i-1]
            y[i] = y[i-1] - 1
            
    return x, y

distance_squared = np.zeros(njumps)

for exp in range(nexperiments):
    x, y = perform_exp()
    
    for i in range(njumps):
        distance_squared[i] = distance_squared[i] + (x[i]**2 + y[i]**2)

distance_squared = distance_squared/nexperiments
            
plt.plot(distance_squared)
plt.xlabel('njumps')
plt.ylabel('msd')
plt.grid()
plt.show()






