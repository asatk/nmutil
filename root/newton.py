import numpy as np

from . import RootFinder, RootSolution

class NewtonRaphson(RootFinder):
    """
    Implements the Newton-Raphson method for finding roots.
    """

    def __init__(funcs,
                 inv_jac,
                 initc,
                 tol: float=1e-5,
                 termc=None,
                 fargs: list=None,
                 callbacks: list=None):
        super().__init__(funcs,
                         initc,
                         tol,
                         termc=termc,
                         fargs=fargs,
                         callbacks=callbacks)
        self._inv_jac = inv_jac
        self._s = RootSolution(initc)


    def step(self):
        s = self._s
        x = self._s.curr()
        fx = self.func(x)
        invjacx = self._inv_jac(x)
        dx = - np.squeeze(np.matmul(fx, invjacx))
        s.step(x + dx)
