import abc
import numpy as np
from typing import Literal

class FFT(metaclass=abc.ABCMeta):

    def __init__(self,
                 n: int,
                 dt: float=1.0,
                 real: bool=False,
                 norm: tuple=None):

        self._n = np.int64(n)
        self._dt = dt
        self._real = real
        self._norm = (1, -1) if norm is None else norm
        self._ft = None

    @abc.abstractmethod
    def _fft_helper(self,
                    data: np.ndarray,
                    isign: int=1,
                    **kwargs) -> np.ndarray:
        """
        Fast Fourier Transform implementation helper function.
        """
        ...


    def fft(self,
            data: np.ndarray,
            **kwargs) -> np.ndarray:
        """
        Fourier Transform
        """
        ft = self._fft_helper(data, isign=1, **kwargs)
        ft *= self._n ** ((1 - self._norm[0])/-2)
        self._ft = ft
        return ft


    def ifft(self,
             data: np.ndarray,
             **kwargs) -> np.ndarray:
        """
        Inverse Fourier Transform
        """
        ift = self._fft_helper(data, isign=-1, **kwargs)
        ift *= self._n ** ((1 - self._norm[1])/-2)
        self._ft = data
        return ift


    def power(self, data: np.ndarray=None) -> np.ndarray:
        """
        One-sided power spectral density
        """

        if self._ft is None and data is None:
            raise ValueError("Argument `data` must be provided if no data "+\
                    "have been transformed yet.")
        elif self._ft is None and data is not None:
            ft = self.fft(data)
        else:
            ft = self._ft

        if self._real:
            ps = 2 * np.abs(ft) ** 2
            return ps
        
        ft_pos = ft[ft >= 0]
        ft_neg = ft[ft < 0]

        return np.abs(ft_pos) ** 2 + np.abs(ft_neg) ** 2

    
    def freq(self) -> np.ndarray:
        """
        Frequencies sampled by FFT
        """

        n = self._n
        nums = np.roll(np.arange(n), (n+1)//2) - n//2
        freqs = nums / self._dt / n

        if self._real:
            return np.abs(freqs[:n//2+1])
        else:
            return freqs
        
        
    def period(self):
        """
        Periods sampled by FFT
        """

