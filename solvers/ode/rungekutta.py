import abc
import numpy as np
from . import DEFAULT_NSTEP, MAX_NSTEP, MAX_ADAPTS
from .solution import Solution
from .ode import ODESolverBase

class RungeKutta4(ODESolverBase):
    """
    Fourth-order Runge Kutta ODE Solver
    """

    def _rk4_helper(self):
        s = self._soln
        i = s.ind()
        sol = np.copy(s[i])
        dt = self._dt
        time = s.time()
        args = self._args
        
        x1 = np.copy(s[i])
        x2 = np.copy(s[i])
        x3 = np.copy(s[i])
        x4 = np.copy(s[i])

        # Stage 1
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(x1, time, *args[j])
            sol[j] += feval / 6
            x2[j] += feval / 2
        
        # Stage 2
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(x2, time + dt / 2, *args[j])
            sol[j] += feval / 3
            x3[j] += feval / 2

        # Stage 3
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(x3, time + dt / 2, *args[j])
            sol[j] += feval / 3
            x4[j] += feval

        # Stage 4
        for j in range(self._order):
            f = self._fns[j]
            feval = dt * f(x4, time + dt, *args[j])
            sol[j] += feval / 6
        return sol

    def step(self):
        sol = self._rk4_helper()
        self._soln.step(sol)


class RungeKutta4Ada(RungeKutta4):
    """
    Fourth-order Adaptive Runge Kutta ODE Solver
    """

    safe1: float=0.9
    safe2: float=4.0
    err: float=1e-2
    eps: float=np.spacing(1)

    def step(self):
        s = self._soln
        dt = self._dt
        for _ in range(MAX_ADAPTS):
            
            # Long step
            super().step()
            sol1 = s[s.ind()]
            s._i -= 1    # Rewind

            # Short steps
            self._dt = dt / 2
            s.set_dt(dt / 2)
            super().step()
            sol2 = super()._rk4_helper()
            s._i -= 1    # Rewind

            deltac = np.max(np.abs(sol1 - sol2))
            deltai = self.err * np.mean((np.abs(sol1), np.abs(sol2)))

            ratio = np.power((deltai + self.eps) / deltac, 1/5)

            dt = np.clip(self.safe1 * dt * ratio,
                         a_min=dt / self.safe2,
                         a_max=dt * self.safe2)
            self._dt = dt
            s.set_dt(dt)

            if ratio < 1:
                break

        super().step()


class RungeKutta2(ODESolverBase):
    """
    Second-order Runge Kutta ODE Solver
    """
    def __init__(
        self,
        funcs,
        nstep: int,
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
