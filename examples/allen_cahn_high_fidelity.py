# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:28:03 2024

@author: mhstr
"""

import numpy as np
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph


def run_high_fidelity(dt, dx, max_t, min_x, max_x):

    # initialize the equation and the space
    eq = PDE({"φ": "0.0001*laplace(φ) + 5 * φ - 5 * φ**3"})
    grid = CartesianGrid([[min_x, max_x]], [int((max_x - min_x) / dx)], periodic=True)
    state = ScalarField.from_expression(grid, "x**2 * cos(pi*x)")
    
    # solve the equation and store the trajectory
    storage = MemoryStorage()
    eq.solve(state, t_range=max_t, solver="scipy", tracker=storage.tracker(dt))
    
    t = np.linspace(0, max_t, int(max_t / dt))
    x = np.linspace(min_x, max_x, int((max_x - min_x) / dx))
    
    return t, x, storage.data
