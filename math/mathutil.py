"""
Utility functions

Anthony Atkinson

2025.02.12
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def interp(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Compute the Lagrange polynomial interpolation from data `x` and `y` for
    sample points in the domain `t`
    """

    # Accumulate the sum over all xi
    term = 0.0
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]

        # Accumulate the product for each xj
        factor = 1.0
        for j in range(len(x)):
            if j == i:
                continue
            xj = x[j]
            factor *= (t - xj) / (xi - xj)
        term += factor * yi
    return term


def cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Get the magnitude of the cross product vector normal to the plane of the
    two co-planar vectors.
    """
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
