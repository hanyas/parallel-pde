from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.linearization import extended
from parsmooth.methods import filter_smoother, iterated_smoothing
from burgers_cole_hopf import run_ch

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cuda")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


class PDE(NamedTuple):
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
    pde: PDE, x: jax.Array, q: float = 1.0, sigma: float = 1000.0, nell: float = 1.0
):
    dt = pde.dt
    dx = pde.dx

    # IWP-2 prior.
    A_dt = jnp.array([[1,dt,dt**2/2],[0,1,dt],[0,0,1]])
    Q_dt = q * jnp.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])

    J = len(x) - 1
    I_Jm1 = jnp.eye(J - 1)
    ell = nell * dx
    C_Jm1 = construct_gram_matrix(x[1:-1], squared_exponential_kernel, sigma, ell)

    A = jnp.kron(A_dt, I_Jm1)
    Q = jnp.kron(Q_dt, C_Jm1)
    return FunctionalModel(lambda u: A @ u, MVNStandard(jnp.zeros(Q.shape[0]), Q))


def pseudo_observation_fn(pde: PDE, x: jax.Array):
    """This is specific for Burgers' equation."""
    J = int(len(x) / 3) + 1
    u = x[: J - 1]
    u_jp1 = jnp.concat((u[1:], jnp.array([pde.u_b])))
    u_jm1 = jnp.concat((jnp.array([pde.u_a]), u[:-1]))
    f = -4.0 * (u_jp1 - 2.*u + u_jm1) / (dx * dx)
    du = x[2 * (J-1):]
    return du + f


def solve_pde(pde: PDE, n_iter: int = 1):
    x, t = construct_grid(pde)
    J = len(x) - 1
    N = len(t) - 1

    # Specify the initial mean and covariance.
    us = pde.u_0(x[1:-1])
    dus = jnp.zeros(J - 1)
    ddus = jnp.zeros(J - 1)
    m0 = jnp.concat((us, dus, ddus))
    P0 = jax.scipy.linalg.block_diag(jnp.zeros((J - 1, J - 1)), jnp.zeros((J - 1, J - 1)), jnp.zeros((J - 1, J - 1)))
    s0 = MVNStandard(m0, P0)

    # Get the transition and observation models.
    transition_model = get_transition_model(pde, x)
    observation_model = FunctionalModel(
        lambda s: pseudo_observation_fn(pde, s), MVNStandard(jnp.zeros(J - 1), 0.0)
    )
    observations = jnp.zeros((N, J - 1))
    init_nominal_trajectory = filter_smoother(
        observations, s0, transition_model, observation_model, extended, None, False
    )
    smoothed_trajectory = iterated_smoothing(
        observations,
        s0,
        transition_model,
        observation_model,
        extended,
        init_nominal_trajectory,
        criterion=lambda i, *_: i < n_iter + 1,
        parallel=False,
        return_loglikelihood=False,
    )
    return t, x, smoothed_trajectory


if __name__ == "__main__":
    
    T = 1.0
    a = 0
    b = 1

    def u_0(x):
        return jnp.sin(jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)
    
    def u_exact(x,t):
        return jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * t) + 0.5 * jnp.sin(4 * jnp.pi * x) * jnp.cos(8 * jnp.pi * t)
    
    exact_x = jnp.linspace(a, b, 1000)

    dx = 0.01
    dt = 0.001
    wave = PDE(a, b, 0.0, 0.0, u_0, T, dx, dt)
    t, x, smoothed_trajectory = solve_pde(wave)

    # Get the posterior mean and variance of the solution.
    J = len(x) - 1
    H_s = jnp.kron(jnp.array([1.0, 0.0, 0.0]), jnp.eye(J - 1))
    us = jnp.einsum("ij, tj -> ti", H_s, smoothed_trajectory.mean)
    Ps = jnp.einsum("ij, tjk, kl -> til", H_s, smoothed_trajectory.cov, H_s.T)
    Ps = jnp.diagonal(Ps, axis1=1, axis2=2)

    # Plotting code from Simo.
    # Frame have to be adjusted according to the discretization.
    ps_frames = [0, int(0.15*T/dt), int(0.5*T/dt), int(0.85*T/dt)]
    exact_frames = [0, 0.15, 0.5, 0.85]
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
        ax.plot(exact_x, u_exact(exact_x,exact_frames[idx]), 'k--', linewidth = 2.5, label = 'Exact')
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(t[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
        ax.set_ylim([-1.55,1.55])
    plt.show()
