import numpy as np

from .root import RootFinderBase, RootSolution

class NewtonRaphson(RootFinderBase):
    """
    Implements the Newton-Raphson method for finding roots.
    """

    def __init__(self,
                 funcs,
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
        self._soln = RootSolution(initc)


    def step(self):
        s = self._soln
        x = self._soln.curr()
        f = self._fns[0]
        fx = f(x)
        invjacx = self._inv_jac(x)
        dx = - np.squeeze(np.matmul([fx], [invjacx]))
        s.step(x + dx)
