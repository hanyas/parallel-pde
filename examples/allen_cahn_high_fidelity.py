# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:28:03 2024

@author: mhstr
"""

import numpy as np
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph


def run_high_fidelity(dt, dx, t, a, b):

    # initialize the equation and the space
    eq = PDE({"φ": "0.0001*laplace(φ) + 5 * φ - 5 * φ**3"})
    grid = CartesianGrid([[a, b]], [int((b - a) / dx)], periodic=True)
    state = ScalarField.from_expression(grid, "x**2 * cos(pi*x)")
    
    # solve the equation and store the trajectory
    storage = MemoryStorage()
    eq.solve(state, t_range=t, solver="scipy", tracker=storage.tracker(dt))
    
    t = np.linspace(0, t, int(t / dt))
    x = np.linspace(a, b, int((b - a) / dx))
    
    return t, x, storage.data
