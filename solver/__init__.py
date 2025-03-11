#~*~# GLOBAL VARIABLES #~*~#
DEFAULT_NSTEP: int = 1000   # Default number of steps for solution. Sets initial size of solution.
MAX_NSTEP: int = 100_000    # Maximum number of steps a solutions can take; prevents memory crash.
MAX_ADAPTS: int = 100       # Maximum number of attempts to find an optimal timestep in adaptive methods

from .solver import SolverBase, SolutionBase

__all__ = [
    "SolutionBase",
    "SolverBase"
]
