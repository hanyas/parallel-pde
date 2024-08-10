import jax
import jax.numpy as jnp

from parpde.objects import PDE, SEParams
from parpde.kernels import squared_exponential
from parpde.utils import get_grid, get_gram_matrix


def burgers_equation(pde: PDE, s: jax.Array):
    flux = lambda u: 0.5 * u**2
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = (flux(u_jp1) - flux(u_jm1)) / (2.0 * pde.dx)
    du = s[J - 1:]
    return du + f


def test_gram_matrix():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    kernel_fn = squared_exponential
    kernel_params = SEParams(length_scale=1.0, signal_stddev=1.0)
    C = get_gram_matrix(x, kernel_fn, kernel_params)
    assert C.shape == (4, 4)
    assert C[0, 0] == 1.0
    assert C[3, 2] == kernel_fn(3.0, 2.0, **kernel_params._asdict())


def test_burgers_equation_fn_derivative():
    flux = lambda u: 0.5 * u**2
    u_0 = lambda u: - jnp.sin(jnp.pi * u)

    dx = 0.1
    dt = 0.05
    t_max = 1.5

    pde = PDE(
        a=-1.0, b=1.0,
        u_a=0.0, u_b=0.0, u_0=u_0,
        t=t_max, dx=dx, dt=dt
    )

    xs, _ = get_grid(pde)
    xs_size = len(xs)

    us = pde.u_0(xs[1:-1])
    dus = jnp.zeros(xs_size - 2)

    m0 = jnp.hstack((us, dus))
    jac = jax.jacfwd(burgers_equation, argnums=1)(pde, m0)
    jac = jac[:, :xs_size - 2]

    assert jnp.allclose(jnp.diagonal(jac, offset=-1), - m0[:xs_size - 3] / (2.0 * dx))
    assert jnp.allclose(jnp.diagonal(jac, offset=1), m0[1:xs_size - 2] / (2.0 * dx))
