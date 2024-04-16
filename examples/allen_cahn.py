# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:05:32 2024

@author: mhstr
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cuda")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


class PDE(NamedTuple):
    gamma_1: float
    gamma_2: float
    a: float
    b: float
    u_a: float
    u_b: float
    u_0: Callable
    T: float
    dx: float
    dt: float


def construct_grid(pde: PDE):
    J = int((pde.b - pde.a) / pde.dx)
    N = int(pde.T / pde.dt)
    x = jnp.linspace(pde.a, pde.b, J + 1)
    t = jnp.linspace(0.0, pde.T, N + 1)
    return x, t


@jax.jit
def squared_exponential_kernel(x1: float, x2: float, sigma: float, ell: float):
    return sigma**2 * jnp.exp(-0.5 * (x1 - x2) ** 2 / ell**2)


@partial(jax.jit, static_argnums=1)
def construct_gram_matrix(x: jax.Array, kernel_fn: Callable, *args):
    return jax.vmap(lambda x1: jax.vmap(lambda x2: kernel_fn(x1, x2, *args))(x))(x)


def get_transition_model(
    pde: PDE, x: jax.Array, q: float = 1.0, sigma: float = 10.0, nell: float = 2.0
):
    dt = pde.dt
    dx = pde.dx

    # IWP-1 prior.
    A_dt = jnp.array([[1, dt], [0, 1]])
    Q_dt = q * jnp.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    J = len(x) - 1
    I_Jm1 = jnp.eye(J - 1)
    ell = nell * dx
    C_Jm1 = construct_gram_matrix(x[1:-1], squared_exponential_kernel, sigma, ell)

    A = jnp.kron(A_dt, I_Jm1)
    Q = jnp.kron(Q_dt, C_Jm1)
    return FunctionalModel(lambda u: A @ u, MVNStandard(jnp.zeros(Q.shape[0]), Q))


def pseudo_observation_fn(pde: PDE, x: jax.Array):
    """This is specific for Burgers' equation."""
    J = int(len(x) / 2) + 1
    u = x[: J - 1]
    u_jp1 = jnp.concat((u[1:], jnp.array([pde.u_b])))
    u_jm1 = jnp.concat((jnp.array([pde.u_a]), u[:-1]))
    f = -pde.gamma_1 * (u_jp1 - 2.*u + u_jm1) / (pde.dx * pde.dx) + pde.gamma_2 * (jnp.power(u,3) - u)
    du = x[J - 1 :]
    return du + f


def solve_pde(pde: PDE, n_iter: int = 1):
    x, t = construct_grid(pde)
    J = len(x) - 1
    N = len(t) - 1

    # Specify the initial mean and covariance.
    us = pde.u_0(x[1:-1])
    dus = jnp.zeros(J - 1)
    m0 = jnp.concat((us, dus))
    P0 = jax.scipy.linalg.block_diag(jnp.zeros((J - 1, J - 1)), jnp.eye(J - 1))
    s0 = MVNStandard(m0, P0)

    # Get the transition and observation models.
    transition_model = get_transition_model(pde, x)
    observation_model = FunctionalModel(
        lambda s: pseudo_observation_fn(pde, s), MVNStandard(jnp.zeros(J - 1), 0.0)
    )
    observations = jnp.zeros((N, J - 1))
    smoothed_trajectory = iterated_smoothing(
        observations,
        s0,
        transition_model,
        observation_model,
        extended,
        criterion=lambda i, *_: i < n_iter + 1,
        parallel=False,
        return_loglikelihood=False,
    )
    return t, x, smoothed_trajectory


if __name__ == "__main__":

    def u_0(u):
        return jnp.power(u,2)*jnp.cos(jnp.pi * u)

    allen_cahn = PDE(0.0001,5.0,-1.0, 1.0, -1.0, -1.0, u_0, 1.0, 0.005, 0.005)
    t, x, smoothed_trajectory = solve_pde(allen_cahn)

    # Get the posterior mean and variance of the solution.
    J = len(x) - 1
    H_s = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(J - 1))
    us = jnp.einsum("ij, tj -> ti", H_s, smoothed_trajectory.mean)
    Ps = jnp.einsum("ij, tjk, kl -> til", H_s, smoothed_trajectory.cov, H_s.T)
    Ps = jnp.diagonal(Ps, axis1=1, axis2=2)

    # Plotting code from Simo.
    # Frame have to be adjusted according to the discretization.
    ps_frames = [0, 50, 150, 200]
    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ps_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.fill_between(
            x[1:-1],
            us[frame_idx, :] - 1.96 * jnp.sqrt(Ps[frame_idx, :]),
            us[frame_idx, :] + 1.96 * jnp.sqrt(Ps[frame_idx, :]),
            label="Confidence",
        )
        ax.plot(x[1:-1], us[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(t[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="lower right")
    plt.show()
