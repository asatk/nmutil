import numpy as np
from typing import Callable

#~*~# GLOBAL VARIABLES #~*~#
DEFAULT_NSTEP: int = 1000   # Default number of steps for solution. Sets initial size of solution.
MAX_NSTEP: int = 100_000    # Maximum number of steps a solutions can take; prevents memory crash.
MAX_ADAPTS: int = 100       # Maximum number of attempts to find an optimal timestep in adaptive methods

# DEFAULT_NSTEP must be an integer
assert(isinstance(DEFAULT_NSTEP, int))
# MAX_NSTEP must be an integer
assert(isinstance(MAX_NSTEP, int))
# MAX_NSTEP must be an integer multiple of DEFAULT_NSTEP
assert(MAX_NSTEP % DEFAULT_NSTEP == 0)

from .ode import ODESolverBase, ODESolution
from .euler import Euler, EulerCromer
from .rungekutta import RungeKutta2, RungeKutta4, RungeKutta4Ada

ODEFunction = Callable[[np.ndarray, np.ndarray, ...], ...]
Callback = Callable[[ODESolution, ...], ...]
TerminatingCondition = Callable[[ODESolution, ...], bool]

__all__ = [
    "DEFAULT_NSTEP",
    "MAX_NSTEP",
    "Euler",
    "EulerCromer",
    "RungeKutta2",
    "RungeKutta4",
    "RungeKutta4Ada",
    "ODESolverBase",
    "ODESolution",
    "Callback",
    "TerminatingCondition"
]
