from typing import Callable, NamedTuple


class PDE(NamedTuple):
    a: float
    b: float
    u_a: float
    u_b: float
    u_0: Callable
    t: float
    dx: float
    dt: float


# Squared Exponential
class SEParams(NamedTuple):
    length_scale: float
    signal_stddev: float


# Stochastic Diff. Eq.
class IWParams(NamedTuple):
    noise_stddev: float
