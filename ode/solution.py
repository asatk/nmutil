import numpy as np

from . import DEFAULT_NSTEP, MAX_NSTEP


class Solution():
    """
    Class containing the data for a solution to a differential equation.
    """

    def __init__(self,
                 nstep: int,
                 order: int,
                 dim: int,
                 dt: float,
                 initc: np.ndarray):

        self._nstep = nstep
        self._order = order
        self._dim = dim
        self._dt = dt

        if nstep == 0:
            self._s = np.zeros((DEFAULT_NSTEP, order, dim))
            self._t = np.zeros(DEFAULT_NSTEP)
        else:
            self._s = np.zeros((nstep, order, dim))
            self._t = np.zeros(nstep)
        self._i = 0
        self._s[0] = initc


    def __getitem__(self, key):
        return self._s[key]

    def __setitem__(self, key, value):
        self._s[key] = value

    def __delete__(self):
        del self._s


    def get_soln(self):
        """
        Returns the entire solution with the step data.
        """
        return self._s[:self._i+1], self._t[:self._i+1]


    def ind(self):
        """
        Returns the current step the solution is on.
        """
        return self._i


    def time(self):
        """
        Returns the latest value of the dependent variable (time).
        """
        return self._t[self._i]


    def set_dt(self, dt: float):
        """
        Sets the step size
        """
        self._dt = dt


    def grow(self) -> bool:
        """
        Increase the size of the solution array by DEFAULT_NSTEP more steps
        """
        shape = self._s.shape
        new_size = shape[0] + DEFAULT_NSTEP
        if new_size > MAX_NSTEP:
            return False
        
        new_shape = [new_size]
        new_shape.extend(shape[1:])

        new_state = np.zeros(new_shape)
        new_state[:shape[0]] = self._s
        self._s = new_state

        new_times = np.zeros(new_size)
        new_times[:shape[0]] = self._t
        self._t = new_times

        return True


    def step(self, data):
        self._i += 1
        self._s[self._i] = data
        self._t[self._i] = self._t[self._i - 1] + self._dt
