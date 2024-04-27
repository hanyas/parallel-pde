# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:51:29 2024

@author: mhstr
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss

# define the initial condition
u_0 = lambda x: -np.sin(np.pi * x)

qn = 128  # order of the quadarture
qx, qw = hermgauss(qn)  # points and weights
nu = 0.005/np.pi        # artificial viscosity
c2 = 2.0 * np.pi * nu


def Cole_Hopf(u, ti, xjs, num_x):
    c1 = 2.0 * np.sqrt(nu * ti)
    for xj_idx in range(num_x):
        top = -np.sum(c1 * qw * np.sin(np.pi * (xjs[xj_idx] - c1 * qx)) * np.exp(-np.cos(np.pi * (xjs[xj_idx] - c1 * qx))/c2))
        bot = np.sum(c1 * qw * np.exp(-np.cos(np.pi * (xjs[xj_idx] - c1 * qx))/c2))
        u[xj_idx] = top/bot
    return u


def run_ch(dt, dx, max_t, min_x, max_x):
    # the number of spatial grid points
    J = int((max_x - min_x) / dx)
    # the number of temporal grid points
    N = int(max_t / dt)
    # construct the mesh in (t, x)
    x = np.linspace(min_x, max_x, J + 1)       # mesh points in space
    t = np.linspace(0, max_t, N + 1)       # mesh points in time
    u_LF = np.zeros((N + 1, J + 1))    # unknown u at new time level for LF
    u_LW = np.zeros((N + 1, J + 1))    # unknown u at new time level for LW
    u_quad = np.zeros((N + 1, J + 1))  # unknown u at new time level for Cole-Hopf
    # Set initial condition
    u_quad[0, :] = u_0(x)
    # advance in time
    for n in range(1, N + 1):
        u_quad[n, :] = Cole_Hopf(u_quad[n, :], t[n], x, J + 1)
        
    return t, x, u_quad