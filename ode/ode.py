import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP
from .solution import Solution

class ODESolverBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for ordinary differential equation solvers.
    """
    
    def __init__(self,
                 funcs: list,
                 nstep: int,
                 order: int,
                 dim: int,
                 dt: float,
                 initc: list,
                 termc=None,
                 fargs: list=None,
                 callbacks: list=None):

        self._fns = funcs
        self._nstep = min(nstep, MAX_NSTEP)
        self._order = order
        self._dim = dim
        self._dt = dt
        self._initc = initc
        self._termc = (lambda _: False) if termc is None else termc
        self._args = [[] for _ in range(order)] if fargs is None else fargs
        self._callbacks = [] if callbacks is None else callbacks
        
        if len(initc) != order:
            raise ValueError(f"Number of initial conditions does not match "+\
                  f"order ({order} != {len(initc)})")

        if len(funcs) != order:
            raise ValueError(f"Order does not match number of functions "+\
                  f"({order} != {len(funcs)})")

        if len(self._args) != order:
            raise ValueError(f"Order does not match number of function "+\
                  f"argument lists ({order} != {len(self._args)})")

        self._soln = Solution(nstep, order, dim, dt, initc)

    
    def __delete__(self):
        del self._soln
    

    @abc.abstractmethod
    def step(self) -> np.ndarray:
        """
        Take a step in the solving routine.
        """
        ...


    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the differential equation.
        """
        fixed_steps = self._nstep > 0
        nsteps = self._nstep if fixed_steps else DEFAULT_NSTEP

        while True:
            for _ in range(nsteps - 1):
                self.step()
                soln = self._soln
                if self._termc(soln):
                    return soln
                for c in self._callbacks:
                    c(soln)
            if fixed_steps:
                break
            else:
                if not self._soln.grow():
                    break

        return self._soln
