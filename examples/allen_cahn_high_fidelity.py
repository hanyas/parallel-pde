from jax import numpy as jnp

import numpy as np
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField


def run_high_fidelity(dt, dx, t, a, b):

    # initialize the equation and the space
    eq = PDE({"φ": "0.005 * laplace(φ) + 5 * φ - 5 * φ**3"})
    grid = CartesianGrid([[a, b]], [int((b - a) / dx)], periodic=True)
    state = ScalarField.from_expression(grid, "x**2 * cos(pi*x)")
    
    # solve the equation and store the trajectory
    storage = MemoryStorage()
    eq.solve(state, t_range=t, solver="scipy", tracker=storage.tracker(dt))

    xs = np.linspace(a, b, int((b - a) / dx))
    ts = np.linspace(0, t, int(t / dt))

    return jnp.asarray(xs), jnp.asarray(ts), jnp.asarray(storage.data)