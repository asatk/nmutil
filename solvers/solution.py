import numpy as np

from . import DEFAULT_NSTEP, MAX_NSTEP

class Rank():

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, key):
        return self._data[:, x, :, :]

class Order():

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, key):
        return self._data[:, :, x, :]

class Dimension():
    
    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, key):
        return self._data[..., x]


class Solution():
    """
    Class containing the data for a solution to a differential equation.
    """

    def __init__(self,
                 nstep: int,
                 rank: int,
                 order: int,
                 dim: int,
                 dt: float):

        self._nstep = nstep
        self._rank = rank
        self._order = order
        self._dim = dim

        if nstep == 0:
            self._s = np.zeros((DEFAULT_NSTEP, rank, order, dim))
            self._t = np.zeros(DEFAULT_NSTEP)
        else:
            self._s = np.zeros((nstep, rank, order, dim))
            self._t = np.zeros(nstep)
        self._i = 0

        self.rank = Rank(self._s)
        self.order = Order(self._s)
        self.dim = Dimension(self._s)


    def get_soln():
        """
        Returns the entire solution with the step data.
        """
        return self._s, self._t


    def __getitem__(self, key):
        return self._s[key]

    def __setitem__(self, key, value):
        return self._s[key] = value


    def ind(self):
        """
        Returns the current step the solution is on.
        """
        return self._i


    def set_dim(self, data, ind: int|list[int]=None):
        if ind is None:
            ind = self._i
        self._s[ind] = data


    def set_dt(self, dt: float):
        """
        Sets the step size
        """
        self._dt = dt


    def grow_soln(self) -> bool:
        """
        Increase the size of the solution array by DEFAULT_NSTEP more steps
        """
        shape = self._s.shape
        new_size = shape[0] + DEFAULT_NSTEP
        if new_size > MAX_NSTEP:
            return False
        
        new_shape = [new_size]
        if len(shape) > 1:
            new_shape.extend(shape[1:])

        new_state = np.zeros(new_shape)
        new_state[:shape[0]] = self._s
        self._s = new_state
        return True


    def step(self, data):
        self._s[self._i] = data
        self._t[self._i] = self._t[self._i - 1] += self._dt
        self._i += 1
