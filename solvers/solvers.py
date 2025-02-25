import abc
import numpy as np

DEFAULT_NSTEP: int = 1000
MAX_NSTEP: int = 100_000

class Solver(abc.ABC):
    
    def __init__(self,
                 nstep: int,
                 stepsize: float,
                 initc: dict,
                 termc=None,
                 callbacks: list=None):
        self._nstep = min(nstep, MAX_NSTEP)
        self._tau = stepsize
        self._initc = initc
        self._termc = termc
        self._callbacks = [] if callbacks is None else callbacks
        self.f = None

        if nstep == 0 and (termc is None or termc == {}):
            print("Solver will not terminate: initialization arguments must "+\
                  "satisfy nstep != 0 or termc is not None/{}")


    @abc.abstractmethod
    def step(self, i: int) -> np.ndarray:
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
        

    def _grow_state(self) -> None:
        shape = self.f.shape
        new_shape = [shape[0] + DEFAULT_NSTEP]
        if len(shape) > 1:
            new_shape.extend(shape[1:])

        new_state = np.zeros(new_shape)
        new_state[:shape[0]] = self.f
        self.f = new_state


    def solve(self, verbose: bool=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the differential equation.
        """
        if self._nstep == 0:
            i = 0
            while True:
                if verbose:
                    print(f"Step [{i}]")
                self.step(i)
                i += 1
                if self._termc(self.f, i) or i + 1 >= MAX_NSTEP:
                    break
                if (i + 1) % DEFAULT_NSTEP == 0:
                    self._grow_state()
                for c in self._callbacks:
                    c(self.f, i)
        else:
            for i in range(0, self._nstep-1):
                if verbose:
                    print(f"Step [{i}]")
                self.step(i)
                if self._termc(self.f, i):
                    break
                for c in self._callbacks:
                    c(self.f, i)
        
        steps = self._get_steps(i)

        return self.f[:i+1], steps


class ODESolver(Solver, metaclass=abc.ABCMeta):
    """
    Base class for generic, non-adaptive, uncoupled ODE solver.
    """
    def __init__(
        self,
        func,
        nstep: int,
        order: int,
        dim: int,
        stepsize: float,
        initc: np.ndarray,
        termc: list=None,
        callbacks: list=None):

        super().__init__(nstep,
                         stepsize,
                         initc,
                         termc,
                         callbacks)
        
        if len(initc) != order:
            print(f"Order does not match number of initial conditions "+\
                  f"({order} != {len(initc)})")
            return None

        self._order = order
        self._dim = dim
        self._g = func

        if nstep == 0:
            self.f = np.zeros((DEFAULT_NSTEP, order, dim))
        else:
            self.f = np.zeros((self._nstep, order, dim))
        self.f[0] = initc


    def _get_steps(self, i: int):
        return self._tau * np.arange(i+1)


class RungeKutta4(ODESolver):
    ...


class RungeKutta4Ada(RungeKutta4):
    ...


class RungeKutta2(ODESolver):
    """
    Second-order Runge Kutta ODE Solver
    """
    def __init__(
        self,
        func,
        nstep: int,
        order: int,
        dim: int,
        stepsize: float,
        initc: np.ndarray,
        coefs: tuple|list=None,
        termc: list=None,
        callbacks: list=None):

        super().__init__(nstep,
                         stepsize,
                         initc,
                         termc,
                         callbacks)

        if coefs is None or len(coefs) != 4:
            self._coefs = (1/2, 1/2, 1, 1)
        else:
            self._coefs = coefs
    

    def step(self, i: int):
        f = self.f
        tau = self._tau
        w1 = self._coefs[0]
        w2 = self._coefs[1]
        a = self._coefs[2]
        b = self._coefs[3]
        
        k1 = f[i,-1] + b * tau * self._g(f[i])  #
        f[i+1,-1] = f[i,-1] + w1 * tau * self._g(f[i]) + w2 * tau * self._g(k1)

        k1 = f[i,:-1] + b * tau * f[i,1:] 
        f[i+1]


class EulerSolverBase(ODESolver, metaclass=abc.ABCMeta):
    """
    Abstract base class for order-N Euler-like ODE solvers
    """

    def __init__(
        self,
        func,
        nstep: int,
        order: int,
        dim: int,
        stepsize: float,
        initc: np.ndarray,
        termc: list=None,
        callbacks: list=None):

        super().__init__(func,
                         nstep,
                         order,
                         dim,
                         stepsize,
                         initc,
                         termc,
                         callbacks)
        




class EulerCromerSolver(EulerSolverBase):
    """
    Order-N Euler-Cromer method ODE solver
    """

    def __init__(
        self,
        func,
        nstep: int,
        order: int,
        dim: int,
        stepsize: float,
        initc: np.ndarray,
        termc: list=None,
        callbacks: list=None):

        super().__init__(
                func,
                nstep,
                order,
                dim,
                stepsize,
                initc,
                termc,
                callbacks)
        
    def step(self, i: int):
        f = self.f
        f[i+1,-1] = f[i,-1] + self._tau * self._g(f[i])
        for j in reversed(range(self._order-1)):
            f[i+1,j] = f[i,j] + self._tau * f[i+1,j+1]


class EulerSolver(EulerSolverBase):
    """
    Order-N Euler method ODE solver
    """

    def __init__(
        self,
        func,
        nstep: int,
        order: int,
        dim: int,
        stepsize: float,
        initc: np.ndarray,
        termc: list=None,
        callbacks: list=None):

        super().__init__(
                func,
                nstep,
                order,
                dim,
                stepsize,
                initc,
                termc,
                callbacks)
        
    def step(self, i: int):
        f = self.f
        f[i+1,-1] = f[i,-1] + self._tau * self._g(f[i])
        f[i+1,:-1] = f[i,:-1] + self._tau * f[i,1:]

