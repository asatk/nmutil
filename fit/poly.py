"""
Polynomial fit module.

Anthony Atkinson

2025.03.11
"""

import numpy as np
from numpy import linalg as la


class PolynomialFit():
    """
    Class encapsulating the functionality for making a polynomial fit.
    """

    def __init__(self, order: int=1):

        self._order = order
        self._powers = np.arange(0, self._order + 1)    # pre-compute list of powers
        self._params = np.empty(order + 1)


    def fit(self, data: np.ndarray, resp: np.ndarray, var: np.ndarray=None):
        
        # Predictors divided by variance
        ypred = np.power.outer(data, self._powers)
        var = 1 if var is None else var
        mtx = ypred / var

        # Response divided by variance
        y = resp
        b = y / var
        
        # Calculate fit params
        a = la.inv(mtx.T @ mtx) @ mtx.T @ b

        # Store parameters used in computation
        self._mtx = mtx
        self._b = b
        self._a = a
        return a


    def coefs(self):
        return self._a

    def order(self):
        return self._order


    def predict(self, data: np.ndarray):
        ypred = np.power.outer(data, self._powers).T
        return self._a @ ypred
