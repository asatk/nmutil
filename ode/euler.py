import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP, MAX_ADAPTS
from .ode import ODESolverBase

class EulerCromer(ODESolverBase):
    """
    Order-N Euler-Cromer method ODE solver
    """
        
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
