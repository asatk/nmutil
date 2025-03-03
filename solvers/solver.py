import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP
form . import Solution

class SolverBase(abc.ABC):
    """
    Abstract base class for differential equation solvers.
    """
    
    def __init__(self,
                 funcs: list,
                 nstep: int,
                 rank: int,
                 order: int,
                 dim: int,
                 dt: float,
                 initc: dict,
                 termc=None,
                 fargs: tuple=None,
                 callbacks: list=None):

        self._fns = funcs
        self._nstep = min(nstep, MAX_NSTEP)
        self._rank = rank
        self._order = order
        self._dim = dim
        self._dt = dt
        self._initc = initc
        self._args = () if fargs = None else fargs
        self._callbacks = [] if callbacks is None else callbacks

        if len(initc) != order:
            print(f"Order does not match number of initial conditions "+\
                  f"({order} != {len(initc)})")
            return None

        self._soln = Solution(nstep, rank, order, dim)
        self._soln.step(initc)


    @abc.abstractmethod
    def step(self) -> np.ndarray:
        """
        Take a step in the solving routine.
        """
        ...


    @abc.abstractmethod
    def _get_steps(self, i: int) -> np.ndarray:
        """
        Return the steps in the independent variable.
        """
        ...
        

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the differential equation.
        """
        fixed_steps = self._nstep > 0
        nsteps = DEFAULT_NSTEP if fixed_steps else self._nstep - 1

        while True:
            for _ in range(nsteps):
                data = self.step()
                if self._termc(self._soln):
                    return self._soln
                for c in self._callbacks:
                    c(self._soln)
            if fixed_steps:
                break
            else:
                if not self._soln.grow_soln():
                    break

        return self._soln


class ODESolverBase(SolverBase, metaclass=abc.ABCMeta):
    """
    Base class for generic, non-adaptive, uncoupled ODE solver.
    """
    def __init__(
        self,
        funcs,
        nstep: int,
        rank: int,
        order: int,
        dim: int,
        dt: float,
        initc: np.ndarray,
        termc: list=None,
        fargs: tuple=None,
        callbacks: list=None):

        super().__init__(funcs,
                         nstep,
                         rank,
                         order,
                         dim,
                         dt,
                         initc,
                         termc,
                         fargs,
                         callbacks)


    def _get_steps(self, i: int):
        return self._dt * np.arange(i+1)


class RungeKutta4(ODESolverBase):
    ...


class RungeKutta4Ada(RungeKutta4):
    ...


class RungeKutta2(ODESolverBase):
    """
    Second-order Runge Kutta ODE Solver
    """
    def __init__(
        self,
        funcs,
        nstep: int,
        rank: int,
        order: int,
        dim: int,
        dt: float,
        initc: np.ndarray,
        coefs: tuple|list=None,
        termc: list=None,
        fargs: tuple=None,
        callbacks: list=None):

        super().__init__(funcs,
                         nstep,
                         rank,
                         order,
                         dim,
                         dt,
                         initc,
                         termc,
                         fargs,
                         callbacks)

        if coefs is None or len(coefs) != 4:
            self._coefs = (1/2, 1/2, 1, 1)
        else:
            self._coefs = coefs
    
    
    def step(self, i: int):
        s = self._soln
        dt = self._dt
        
        w1 = self._coefs[0]
        w2 = self._coefs[1]
        a = self._coefs[2]
        b = self._coefs[3]
        
        feval = dt * self._g(s, *self._args)
        temp = s[i,0] + b * feval

        t1 = w1 * feval
        t2 = w2 * dt * self._g(temp, *self._args)

        s[i+1,0] = s[i,0] + t1 + t2


class EulerCromer(ODESolverBase):
    """
    Order-N Euler-Cromer method ODE solver
    """

    def __init__(
        self,
        funcs,
        nstep: int,
        rank: int,
        order: int,
        dim: int,
        dt: float,
        initc: np.ndarray,
        termc: list=None,
        fargs: tuple=None,
        callbacks: list=None):

        super().__init__(
                funcs,
                nstep,
                rank,
                order,
                dim,
                dt,
                initc,
                termc,
                fargs,
                callbacks)
        
    def step(self, i: int):
        s = self._soln
        dt = self._dt

        s[i+1,-1] = s[i,-1] + dt * self._g(s, *self._args)
        for j in reversed(range(self._order - 1)):
            s[i+1,j] = s[i,j] + dt * s[i+1,j+1]


class Euler(ODESolverBase):
    """
    Order-N Euler method ODE solver
    """

    def __init__(
        self,
        funcs,
        nstep: int,
        rank: int,
        order: int,
        dim: int,
        dt: float,
        initc: np.ndarray,
        termc: list=None,
        fargs: tuple=None,
        callbacks: list=None):

        super().__init__(
                funcs,
                nstep,
                rank,
                order,
                dim,
                dt,
                initc,
                termc,
                fargs,
                callbacks)
        
    def step(self, i: int):
        s = self._soln
        dt = self._dt

        fevals = np.apply()

        s[i+1] = s[i] + dt * fevals

        s[i+1,-1] = s[i,-1] + dt * self._g(s, *self._args)
        s[i+1,:-1] = s[i,:-1] + dt * s[i,1:]
