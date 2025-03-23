import numpy as np
from numpy import fft
from typing import Literal

from .fft import FFT

class NumpyFFT(FFT):
    """
    Numpy's implementation of the Cooley-Tukey FFT algorithm.
    """

    def _fft_helper(self,
            data: np.ndarray,
            isign: int=1,
            n: int=None,
            axis: int=-1,
            norm: Literal["backward", "ortho", "forward"]="backward",
            out=None) -> np.ndarray:
        
        if isign == 1:
            if self._real:
                ft = fft.rfft(data, n, axis, norm, out)
            else:
                ft = fft.fft(data, n, axis, norm, out)
        else:
            if self._real:
                ft = fft.irfft(data, n, axis, norm, out)
            else:
                ft = fft.ifft(data, n, axis, norm, out)
        return ft
