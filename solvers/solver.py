import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP
from . import Solution

class SolverBase(abc.ABC):
    """
    Abstract base class for differential equation solvers.
    """
    
    def __init__(self,
                 funcs: list,
                 nstep: int,
                 order: int,
                 rank: int,
                 dim: int,
                 dt: float,
                 initc: dict,
                 termc=None,
                 fargs: list[list]=None,
                 callbacks: list=None):

        self._fns = funcs
        self._nstep = min(nstep, MAX_NSTEP)
        self._order = order
        self._rank = rank
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

        self._soln = Solution(nstep, order, rank, dim, dt, initc[:,None,...])


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


class RungeKutta4(ODESolverBase):
    """
    Fourth-order Runge Kutta ODE Solver
    """

    def step(self):
        s = self._soln
        i = s.ind()
        sol = np.copy(s[i])
        dt = self._dt
        time = s.time()
        args = self._args
        
        # Stage 1
        temp = np.copy(s[i])
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(temp, time, *args[j])
            sol[j] += feval / 6
            temp[j] += feval / 2
        
        # Stage 2
        temp = np.copy(s[i])
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(temp, time + dt / 2, *args[j])
            sol[j] += feval / 3
            temp[j] += feval / 2

        # Stage 3
        temp = np.copy(s[i])
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(temp, time + dt / 2, *args[j])
            sol[j] += feval / 3
            temp[j] += feval

        # Stage 4
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(temp, time + dt, *args[j])
            sol[j] += feval / 6

        s.step(sol)


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
    
    
    def step(self):
        s = self._soln
        i = s.ind()
        sol = np.copy(s[i])
        temp = np.copy(s[i])
        dt = self._dt
        time = s.time()
        args = self._args
        
        w1 = self._coefs[0]
        w2 = self._coefs[1]
        a = self._coefs[2]
        b = self._coefs[3]

        # Stage 1
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(s[i], time, *args[j])
            sol[j] += w1 * feval
            temp[j] += b * feval
        
        # Stage 2
        for j in range(self._order):
            f = self._fns[j]
            sol[j] += w2 * dt * f(temp, time + a * dt, *self._args[j])

        s.step(sol)


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
        
    def step(self):
        s = self._soln
        i = s.ind()
        sol = np.empty_like(s[i])
        dt = self._dt
        time = s.time()
        args = self._args

        sol[-1] = s[i, -1] + dt * self._fns[-1](s[i], time, *args[-1])
        for j in reversed(range(self._order - 1)):
            f = self._fns[j]
            feval = dt * f(sol, time + dt, *args[j])
            sol[j] = s[i,j] + feval

        s.step(sol)


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
        
    def step(self):
        s = self._soln
        i = s.ind()
        sol = np.empty_like(s[i])
        dt = self._dt
        time = s.time()
        args = self._args

        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(s[i], time, *args[j])
            sol[j] = s[i,j] + feval

        s.step(sol)
