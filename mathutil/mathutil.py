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
    Compute a 3-point polynomial fit of the data `x` and `y` at sample points
    in the domain `t`
    """

    t1 = y[0] * (t - x[1]) * (t - x[2]) / (x[0] - x[1]) / (x[0] - x[2])
    t2 = y[1] * (t - x[0]) * (t - x[2]) / (x[1] - x[0]) / (x[1] - x[2])
    t3 = y[2] * (t - x[0]) * (t - x[1]) / (x[2] - x[0]) / (x[2] - x[1])
    return t1 + t2 + t3

def cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
