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
TUTORIAL 8A: Integration using the Monte-Carlo method
Calculates area of circle using Monte-Carlo method and compares with the analytical area.
"""

import numpy as np
import matplotlib.pyplot as plt

mesh_x, mesh_y = 100, 100

num_samples = 1000000
count_in_circle = 0
radius2 = mesh_x**2

for i in range(num_samples):
    x, y = np.random.randint(mesh_x), np.random.randint(mesh_y)
    
    if x**2 + y**2 <= radius2:
        count_in_circle += 1

area_of_curve = count_in_circle*(mesh_x*mesh_y)/num_samples
area_analytical = np.pi*radius2/4

print(area_of_curve, area_analytical)

    
