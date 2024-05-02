import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams, IWParams
from bayes_pde.kernels import squared_exponential
from bayes_pde.kernels import once_integrated_wiener
from bayes_pde.kernels import spatio_temporal

from bayes_pde.utils import get_grid
from bayes_pde.solvers import parallel_solver

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.methods import filter_smoother
from parsmooth.linearization import cubature

import jaxopt


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


def positivity_constraint(x):
    return jnp.log1p(jnp.exp(x))


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cuda")

    dx = 0.005
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

    # Specify prior
    us = pde.u_0(xs[1:-1])
    dus = jnp.zeros(xs_size - 2)

    m0 = jnp.hstack((us, dus))
    P0 = jax.scipy.linalg.block_diag(
        jnp.eye(xs_size - 2) * 1e-8,
        jnp.eye(xs_size - 2)
    )
    prior = MVNStandard(m0, P0)

    # Specify observation model
    observation_model = FunctionalModel(
        lambda s: burgers_equation(pde, s),
        MVNStandard(
            jnp.zeros(xs_size - 2),
            1e-8 * jnp.eye(xs_size - 2)
        )
    )

    observations = jnp.zeros((ts_size - 1, xs_size - 2))

    def log_likelihood_fn(log_params):
        # Transform parameters back to normal space
        length_scale, signal_stddev, noise_stddev = positivity_constraint(log_params)

        # Specify transition model
        spatial_params = SEParams(length_scale=length_scale, signal_stddev=signal_stddev)
        temporal_params = IWParams(noise_stddev=noise_stddev)

        A, Q = spatio_temporal(
            spatial_params,
            temporal_params,
            squared_exponential,
            once_integrated_wiener,
            xs, dx, dt
        )
        transition_model = FunctionalModel(
            lambda u: A @ u,
            MVNStandard(jnp.zeros(Q.shape[0]), Q)
        )

        init_trajectory = filter_smoother(
            observations,
            prior,
            transition_model,
            observation_model,
            cubature,
            None,
            False
        )

        _, ell = parallel_solver(
            observations,
            prior,
            transition_model,
            observation_model,
            init_trajectory,
            cubature,
            nb_iter=25,
            return_loglikelihood=True
        )
        return - 1.0 * ell


    solver = jaxopt.ScipyMinimize(
        fun=log_likelihood_fn,
        method="SLSQP",
        tol=1e-4,
        maxiter=50,
        jit=True,
    )

    init_log_params = jnp.array([0.7, 3.5, -0.75])
    result = solver.run(init_log_params)

    params = positivity_constraint(result.params)
    print("params: ", params)