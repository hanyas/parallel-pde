import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams, SDEParams
from bayes_pde.kernels import squared_exponential
from bayes_pde.kernels import first_order_integrated_wiener
from bayes_pde.kernels import spatio_temporal

from bayes_pde.utils import get_grid
from bayes_pde.solvers import parallel_solver
from burgers_cole_hopf import run_cole_hopf

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.methods import filter_smoother
from parsmooth.linearization import extended

import time
import matplotlib.pyplot as plt


def burgers_equation(pde: PDE, s: jax.Array):
    flux = lambda u: 0.5 * u ** 2
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = (flux(u_jp1) - flux(u_jm1)) / (2.0 * pde.dx)
    du = s[J - 1:]
    return du + f


def u_0(u):
    return - jnp.sin(jnp.pi * u)


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    dx = 0.01
    dt = 0.01
    t_max = 1.0

    pde = PDE(
        a=-1.0, b=1.0,
        u_a=0.0, u_b=0.0, u_0=u_0,
        t=t_max, dx=dx, dt=dt
    )

    xs, ts = get_grid(pde)
    xs_size = len(xs)
    ts_size = len(ts)

    us = pde.u_0(xs[1:-1])
    dus = jnp.zeros(xs_size - 2)

    m0 = jnp.hstack((us, dus))
    P0 = jax.scipy.linalg.block_diag(
        jnp.eye(xs_size - 2) * 0.0,
        jnp.eye(xs_size - 2),
    )
    prior = MVNStandard(m0, P0)

    # Specify transition model
    spatial_params = SEParams(length_scale=2.0, signal_variance=25.0)
    temporal_params = SDEParams(noise_variance=0.5)

    A, Q = spatio_temporal(
        spatial_params,
        temporal_params,
        squared_exponential,
        first_order_integrated_wiener,
        xs, dx, dt
    )
    transition_model = FunctionalModel(
        lambda u: A @ u,
        MVNStandard(jnp.zeros(Q.shape[0]), Q)
    )

    # Specify observation model
    observation_model = FunctionalModel(
        lambda s: burgers_equation(pde, s),
        MVNStandard(
            jnp.zeros(xs_size - 2),
            1e-8 * jnp.eye(xs_size - 2)
        )
    )

    observations = jnp.zeros((ts_size - 1, xs_size - 2))

    init_trajectory = filter_smoother(
        observations, prior, transition_model, observation_model, extended, None, False
    )

    start = time.time()
    smoothed_trajectory = parallel_solver(
        observations,
        prior,
        transition_model,
        observation_model,
        init_trajectory,
        nb_iter=10,
    )
    jax.block_until_ready(smoothed_trajectory)
    end = time.time()
    print("time: ", end - start)

    # Get the posterior mean and variance of the solution
    mask = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(xs_size - 2))
    us_par = jnp.einsum("ij, tj -> ti", mask, smoothed_trajectory.mean)
    Ps_par = jnp.einsum("ij, tjk, kl -> til", mask, smoothed_trajectory.cov, mask.T)
    Ps_par = jnp.diagonal(Ps_par, axis1=1, axis2=2)

    # Frame have to be adjusted according to the discretization
    xs_ch, ts_ch, us_ch = run_cole_hopf(dt, dx, t_max, -1.0, 1.0)
    ch_frames = [0, int(t_max / dt / 3), int(2 * t_max / dt / 3), int(t_max / dt)]
    ieks_frames = [0, int(t_max / dt / 3), int(2 * t_max / dt / 3), int(t_max / dt)]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.fill_between(
            xs[1:-1],
            us_par[frame_idx, :] - 3.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            us_par[frame_idx, :] + 3.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            label="Confidence",
        )
        ax.plot(xs[1:-1], us_par[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        ax.plot(xs_ch, us_ch[ch_frames[idx], :], 'k--', linewidth=2.5, label='CH')
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(ts[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
        ax.set_ylim([-1.55, 1.55])
    plt.show()