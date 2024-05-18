import numpy as np
from numpy.polynomial.hermite import hermgauss

# define the initial condition
u_0 = lambda x: -np.sin(np.pi * x)

qn = 128                # order of the quadarture
qx, qw = hermgauss(qn)  # points and weights
nu = 1e-2  # 0.005 / np.pi      # artificial viscosity
c2 = 2.0 * np.pi * nu


def cole_hopf(u, ti, xjs, num_x):
    c1 = 2.0 * np.sqrt(nu * ti)
    for xj_idx in range(num_x):
        top = -np.sum(c1 * qw * np.sin(np.pi * (xjs[xj_idx] - c1 * qx)) * np.exp(-np.cos(np.pi * (xjs[xj_idx] - c1 * qx))/c2))
        bot = np.sum(c1 * qw * np.exp(-np.cos(np.pi * (xjs[xj_idx] - c1 * qx))/c2))
        u[xj_idx] = top / bot
    return u


def run_cole_hopf(dt, dx, t, a, b):
    # the number of spatial grid points
    xs_size = int((b - a) / dx)
    # the number of temporal grid points
    ts_size = int(t / dt)
    # construct the mesh in (t, x)
    xs = np.linspace(a, b, xs_size + 1)               # mesh points in space
    ts = np.linspace(0, t, ts_size + 1)               # mesh points in time
    u_LF = np.zeros((ts_size + 1, xs_size + 1))       # unknown u at new time level for LF
    u_LW = np.zeros((ts_size + 1, xs_size + 1))       # unknown u at new time level for LW
    u_quad = np.zeros((ts_size + 1, xs_size + 1))     # unknown u at new time level for Cole-Hopf
    # Set initial condition
    u_quad[0, :] = u_0(xs)
    # advance in time
    for n in range(1, ts_size + 1):
        u_quad[n, :] = cole_hopf(u_quad[n, :], ts[n], xs, xs_size + 1)
        
    return xs, ts, u_quad