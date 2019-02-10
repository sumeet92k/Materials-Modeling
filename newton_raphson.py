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
TUTORIAL 3: Newton-Raphson root finding 
Computes the equilibrium phase compositions from the given free-energy curves 
using the Newton-Raphson root finding technique
"""

import numpy as np
import matplotlib.pyplot as plt

# the free energies of the phases
def f(x):
    return 0.5*x*x + 0.5*x - 0.25
    
def g(x):
    return 0.5*x*x

# the slope of the free energies
def f_prime(x):
    return x + 0.5

def g_prime(x):
    return x

# one of the chemical potentials i.e. the intercept with the x=0 axis
def f_intercept(x):
    return -0.5*x*x - 0.25

def g_intercept(x):
    return -0.5*x*x

# we have to find the root of the functions F() and G()
def F(x1, x2):
    return f_prime(x1) - g_prime(x2)

def G(x1, x2):
    return f_intercept(x1) - g_intercept(x2)

# calculating the elements inside the jacobian matrix
def f_2_prime(x):
    return 1

def g_2_prime(x):
    return 1

def f_int_prime(x):
    return -x

def g_int_prime(x):
    return -x

def jacobian(x1, x2):
    return np.array( [ [ f_2_prime(x1), -g_2_prime(x2) ], [ f_int_prime(x1), -g_int_prime(x2) ] ] )

x_array = np.linspace(0, 1, 21)

# the initial guess
x_init = np.array([100.1, 100]) # if x_init[0]=x_init[1] will create a singular matrix
iterations = 10 
x = x_init

for i in range(iterations):
    inv_jacobian = np.linalg.inv( jacobian( x[0], x[1]) )
    
    x = x - np.dot(inv_jacobian, np.array( [ F(x[0], x[1]), G(x[0], x[1]) ] ) )

    print(i, x) # iteration number and roots

# for plotting the free energy curves f() and g()
plt.plot(x_array, f(x_array), label='f')
plt.plot(x_array, g(x_array), label='g')
plt.legend()
plt.show()