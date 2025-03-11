import numpy as np
from numpy import linalg as la

from ..solver import SolverBase, SolutionBase

class RootSolution: ...


def root_tol_check(s: RootSolution, tol: float=1e-3):
    return la.norm(s.prev() - s.curr()) / la.norm(s.curr()) < tol


class RootSolution(SolutionBase):
    """
    Class encapsulating the necessary information for a root-finder.
    """
    def __init__(self, initc):
        super().__init__()
        self._initc = initc
        self._c = np.copy(initc)
        self._p = np.copy(initc)


    def get_soln(self):
        return self._c


    def curr(self):
        return self._c


    def prev(self):
        return self._p

    def step(self, x):
        self._p = self._c
        self._c = x


    def grow(self):
        pass


class RootFinderBase(SolverBase):
    """
    Abstract base class for root finding methods.
    """

    def __init__(self,
                 funcs: list,
                 initc: list,
                 tol: float=1e-3,
                 termc=None,
                 fargs: list=None,
                 callbacks: list=None):
        if termc is None:
            termc_ = lambda s: root_tol_check(s, tol)
        else:
            termc_ = lambda s: (root_tol_check(s, tol) | termc(s))
        super().__init__(
                funcs,
                0,
                initc,
                termc=termc_,
                fargs=fargs,
                callbacks=callbacks)
        self._tol = tol
