import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP

class SolutionBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for solutions from numerical solvers.
    """

    @abc.abstractmethod
    def get_soln(self):
        """
        Returns the solution.
        """
        ...


    @abc.abstractmethod
    def step(self, x):
        """
        Update the solution by one step.
        """
        ...


    @abc.abstractmethod
    def grow(self):
        """
        Grow the data area containing the solution.
        """
        ...


class SolverBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for numerical solvers.
    """
    
    def __init__(self,
                 funcs: list,
                 nstep: int,
                 initc: list,
                 termc=None,
                 fargs: list=None,
                 callbacks: list=None):

        self._fns = funcs if isinstance(funcs, list) else [funcs]
        self._nstep = min(nstep, MAX_NSTEP)
        self._initc = initc
        self._termc = (lambda _: False) if termc is None else termc
        self._args = [[] for _ in range(len(self._fns))] if fargs is None else fargs
        self._callbacks = [] if callbacks is None else callbacks

    
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
