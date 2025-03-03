import numpy as np
from typing import Callable

#~*~# GLOBAL VARIABLES #~*~#
DEFAULT_NSTEP: int = 1000   # Default number of steps for solution. Sets initial size of solution.
MAX_NSTEP: int = 100_000    # Maximum number of steps a solutions can take; prevents memory crash.

# DEFAULT_NSTEP must be an integer
assert(isinstance(DEFAULT_NSTEP, int))
# MAX_NSTEP must be an integer
assert(isinstance(MAX_NSTEP, int))
# MAX_NSTEP must be an integer multiple of DEFAULT_NSTEP
assert(MAX_NSTEP % DEFAULT_NSTEP == 0)

from .solution import Solution
from .solver import Euler, EulerCromer, RungeKutta2

ODEFunction = Callable[[np.ndarray, np.ndarray, ...], ...]
Callback = Callable[[Solution, ...], ...]
TerminatingCondition = Callable[[Solution, ...], bool]

__all__ = [
    "DEFAULT_NSTEP",
    "MAX_NSTEP",
    "Euler",
    "EulerCromer",
    "RungeKutta2",
    "Solution",
    "Callback",
    "TerminatingCondition"
]
