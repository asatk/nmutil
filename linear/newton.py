import numpy as np


class Solution():
    def __init__(order: int,
                 dim: int,
                 initc):
        if initc.shape != (order, dim):
            print("Shape of initial condition does not match expectation:"+\
                  f"{initc.shape} != {(order, dim)}")
        self._s = initc

    def step(self):
        self._s


class LinearSolver():
    def __init__(order: int,
                 dim: int,
                 tol: float=1e-5,
                 initc):
        self._fns = funcs
        self._order = order
        self._dim = dim
        self._tol = tol
        self._soln = Solution(order, dim, dx)

    def solve(self):

        err = tol + 1
        while err > tol:
            self.step()





class Newton(LinearSolver):
    
    def __init__(func,
                 inv_jac,
                 order: int,
                 dim: int,
                 tol: float,
                 initc):
        super().__init__(order,
                         dim,
                         tol,
                         initc)
        self._inv_jac = inv_jac

    def step(self):
        s = self._soln
        fx = self.func(s)
        jacx = self._inv_jac(s)
        dx = 
        s.step()


