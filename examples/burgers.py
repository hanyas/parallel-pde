from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.linearization import extended
from parsmooth.methods import filter_smoother, iterated_smoothing

from burgers_cole_hopf import run_cole_hopf

import time
import matplotlib.pyplot as plt


class PDE(NamedTuple):
    a: float
    b: float
    u_a: float
    u_b: float
    u_0: Callable
    T: float
    dx: float
    dt: float
    flux: Callable


def construct_grid(pde: PDE):
    J = int((pde.b - pde.a) / pde.dx)
    N = int(pde.T / pde.dt)
    x_grid = jnp.linspace(pde.a, pde.b, J + 1)
    t_grid = jnp.linspace(0.0, pde.T, N + 1)
    return x_grid, t_grid


@jax.jit
def squared_exponential_kernel(x1: float, x2: float, sigma: float, ell: float):
    return sigma**2 * jnp.exp(-0.5 * (x1 - x2) ** 2 / ell**2)


@partial(jax.jit, static_argnums=1)
def construct_gram_matrix(x: jax.Array, kernel_fn: Callable, *args):
    return jax.vmap(lambda x1: jax.vmap(lambda x2: kernel_fn(x1, x2, *args))(x))(x)


def get_transition_model(
    pde: PDE, x_grid: jax.Array, q: float = 0.5,
    sigma: float = 25.0, nell: float = 2.0
):
    dt = pde.dt
    dx = pde.dx

    # IWP-1 prior.
    A_dt = jnp.array(
        [
            [1, dt],
            [0, 1]
        ]
    )
    Q_dt = q * jnp.array(
        [
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt]
        ]
    )

    ell = nell * dx
    C_Jm1 = construct_gram_matrix(x_grid[1:-1], squared_exponential_kernel, sigma, ell)

    J = len(x_grid) - 1
    I_Jm1 = jnp.eye(J - 1)

    A = jnp.kron(A_dt, I_Jm1)
    Q = jnp.kron(Q_dt, C_Jm1)
    return FunctionalModel(
        lambda u: A @ u,
        MVNStandard(jnp.zeros(Q.shape[0]), Q)
    )


def burgers_equation_fn(pde: PDE, s: jax.Array):
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.concat((u[1:], jnp.array([pde.u_b])))
    u_jm1 = jnp.concat((jnp.array([pde.u_a]), u[:-1]))
    f = (pde.flux(u_jp1) - pde.flux(u_jm1)) / (2.0 * pde.dx)
    du = s[J - 1:]
    return du + f


def flux(u):
    return 0.5 * u**2


def u_0(u):
    return -jnp.sin(jnp.pi * u)


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    max_t = 1.0
    min_x = -1.0
    max_x = 1.0

    dx = 0.01
    dt = 0.01

    t_grid_ch, x_grid_ch, us_ch = run_cole_hopf(dt, dx, max_t, min_x, max_x)

    pde = PDE(min_x, max_x, 0.0, 0.0, u_0, max_t, dx, dt, flux)

    x_grid, t_grid = construct_grid(pde)
    x_size = len(x_grid) - 1
    t_size = len(t_grid) - 1

    us = pde.u_0(x_grid[1:-1])
    dus = jnp.zeros(x_size - 1)

    m0 = jnp.concat((us, dus))
    P0 = 1e-2 * jax.scipy.linalg.block_diag(
        jnp.eye(x_size - 1) * 0.0,
        jnp.eye(x_size - 1),
    )
    prior = MVNStandard(m0, P0)

    transition_model = get_transition_model(pde, x_grid)

    observation_model = FunctionalModel(
        lambda s: burgers_equation_fn(pde, s),
        MVNStandard(jnp.zeros(x_size - 1), jnp.eye(x_size - 1) * 1e-8)
    )

    observations = jnp.zeros((t_size, x_size - 1))

    init_nominal_trajectory = filter_smoother(
        observations, prior, transition_model, observation_model, extended, None, False
    )

    nb_iter = 10

    @jax.jit
    def par_solver(init_trajectory, nb_iter):
        par_res = iterated_smoothing(
            observations,
            prior,
            transition_model,
            observation_model,
            extended,
            init_trajectory,
            parallel=True,
            criterion=lambda i, *_: i < nb_iter,
            return_loglikelihood=False,
        )
        return par_res


    @jax.jit
    def seq_solver(init_trajectory, nb_iter):
        seq_res = iterated_smoothing(
            observations,
            prior,
            transition_model,
            observation_model,
            extended,
            init_trajectory,
            parallel=False,
            criterion=lambda i, *_: i < nb_iter,
            return_loglikelihood=False,
        )
        return seq_res


    smoothed_trajectory = par_solver(init_nominal_trajectory, nb_iter)
    jax.block_until_ready(smoothed_trajectory)

    start = time.time()
    smoothed_trajectory = par_solver(init_nominal_trajectory, nb_iter)
    jax.block_until_ready(smoothed_trajectory)
    end = time.time()
    print("time: ", end - start)

    # Get the posterior mean and variance of the solution
    mask = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(x_size - 1))
    us_par = jnp.einsum("ij, tj -> ti", mask, smoothed_trajectory.mean)
    Ps_par = jnp.einsum("ij, tjk, kl -> til", mask, smoothed_trajectory.cov, mask.T)
    Ps_par = jnp.diagonal(Ps_par, axis1=1, axis2=2)

    # Frame have to be adjusted according to the discretization
    ieks_frames = [0, int(max_t / dt / 3), int(2 * max_t / dt / 3), int(max_t / dt)]
    ch_frames = [0, int(max_t / dt / 3), int(2 * max_t / dt / 3), int(max_t / dt)]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.fill_between(
            x_grid[1:-1],
            us_par[frame_idx, :] - 3.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            us_par[frame_idx, :] + 3.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            label="Confidence",
        )
        ax.plot(x_grid[1:-1], us_par[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        ax.plot(x_grid_ch, us_ch[ch_frames[idx], :], 'k--', linewidth=2.5, label='CH')
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(t_grid[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
    plt.show()

    smoothed_trajectory = seq_solver(init_nominal_trajectory, nb_iter)
    jax.block_until_ready(smoothed_trajectory)

    start = time.time()
    smoothed_trajectory = seq_solver(init_nominal_trajectory, nb_iter)
    jax.block_until_ready(smoothed_trajectory)
    end = time.time()
    print("time: ", end - start)

    # Get the posterior mean and variance of the solution
    mask = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(x_size - 1))
    us_seq = jnp.einsum("ij, tj -> ti", mask, smoothed_trajectory.mean)
    Ps_seq = jnp.einsum("ij, tjk, kl -> til", mask, smoothed_trajectory.cov, mask.T)
    Ps_seq = jnp.diagonal(Ps_seq, axis1=1, axis2=2)

    # Frame have to be adjusted according to the discretization
    ieks_frames = [0, int(max_t / dt / 3), int(2 * max_t / dt / 3), int(max_t / dt)]
    ch_frames = [0, int(max_t / dt / 3), int(2 * max_t / dt / 3), int(max_t / dt)]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.fill_between(
            x_grid[1:-1],
            us_seq[frame_idx, :] - 3.0 * jnp.sqrt(Ps_seq[frame_idx, :]),
            us_seq[frame_idx, :] + 3.0 * jnp.sqrt(Ps_seq[frame_idx, :]),
            label="Confidence",
        )
        ax.plot(x_grid[1:-1], us_seq[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        ax.plot(x_grid_ch, us_ch[ch_frames[idx], :], 'k--', linewidth=2.5, label='CH')
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(t_grid[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
    plt.show()