import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams, IWParams
from bayes_pde.kernels import squared_exponential
from bayes_pde.kernels import twice_integrated_wiener
from bayes_pde.kernels import spatio_temporal

from bayes_pde.utils import get_grid
from bayes_pde.solvers import batch_solver

from newton_smoothers.base import FunctionalModel, MVNStandard
from newton_smoothers.utils import none_or_concat

import time
import matplotlib.pyplot as plt


def wave_equation(pde: PDE, s: jax.Array):
    J = int(len(s) / 3) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = -4.0 * (u_jp1 - 2. * u + u_jm1) / (pde.dx * pde.dx)
    du = s[2 * (J - 1):]
    return du + f


def u_0(x):
    return jnp.sin(jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    dx = 0.1
    dt = 0.01
    t_max = 1.0

    pde = PDE(
        a=0.0, b=1.0,
        u_a=0.0, u_b=0.0, u_0=u_0,
        t=t_max, dx=dx, dt=dt
    )

    xs, ts = get_grid(pde)
    xs_size = len(xs)
    ts_size = len(ts)

    # Specify prior
    us = pde.u_0(xs[1:-1])
    dus = jnp.zeros(xs_size - 2)
    ddus = jnp.zeros(xs_size - 2)

    m0 = jnp.hstack((us, dus, ddus))
    P0 = jax.scipy.linalg.block_diag(
        jnp.eye(xs_size - 2) * 1e-8,
        jnp.eye(xs_size - 2),
        jnp.eye(xs_size - 2)
    )
    prior = MVNStandard(m0, P0)

    # Specify transition model
    spatial_params = SEParams(length_scale=1.0, signal_variance=1000.0)
    temporal_params = IWParams(noise_variance=10.0)

    A, Q = spatio_temporal(
        spatial_params,
        temporal_params,
        squared_exponential,
        twice_integrated_wiener,
        xs, dx, dt
    )
    transition_model = FunctionalModel(
        lambda u: A @ u,
        MVNStandard(jnp.zeros(Q.shape[0]), Q)
    )

    # Specify observation model
    observation_model = FunctionalModel(
        lambda s: wave_equation(pde, s),
        MVNStandard(
            jnp.zeros(xs_size - 2),
            1e-8 * jnp.eye(xs_size - 2)
        )
    )

    observations = jnp.zeros((ts_size - 1, xs_size - 2))

    init_trajectory = MVNStandard(
        mean=jnp.zeros((ts_size - 1, 3 * (xs_size - 2))),
        cov=jnp.repeat(
            jnp.eye(3 * (xs_size - 2))[jnp.newaxis, ...],
            ts_size - 1, axis=0
        ),
    )
    init_trajectory = none_or_concat(init_trajectory, prior, 1)

    start = time.time()
    batch_trajectory = batch_solver(
        observations,
        prior,
        transition_model,
        observation_model,
        init_trajectory,
        nb_iter=10,
    )
    jax.block_until_ready(batch_trajectory)
    end = time.time()
    print("time: ", end - start)

    # Get the posterior mean and variance of the solution
    mask = jnp.kron(jnp.array([1.0, 0.0, 0.0]), jnp.eye(xs_size - 2))
    us_btc = jnp.einsum("ij, tj -> ti", mask, batch_trajectory)

    # Frame have to be adjusted according to the discretization
    ieks_frames = [0, int(0.15 * t_max / dt), int(0.5 * t_max / dt), int(0.85 * t_max / dt)]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.plot(xs[1:-1], us_btc[frame_idx, :], "b-", linewidth=2.0, label="IEKS")
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(ts[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
        ax.set_ylim([-1.55, 1.55])
    plt.show()