import abc
import numpy as np
from typing import Literal

class FFT(metaclass=abc.ABCMeta):

    def __init__(self,
                 domain: Literal["time", "freq"]="time",
                 spacing: float=1.0):

        self._domain = domain
        self._spacing = spacing
        self._isreal = np.isrealobj(data)
        self._ft = None if domain == "time" else data

    @abc.abstractmethod
    def _fft_helper(self,
                    data: np.ndarray,
                    isign: int=1,
                    *args) -> np.ndarray:
        """
        Fast Fourier Transform implementation helper function.
        """
        ...


    def fft(self,
            data: np.ndarray,
            isign: int=1, *kwargs) -> np.ndarray:
        """
        Fourier Transform
        """
        ft = self._fft_helper(data, isign, *kwargs)
        self._ft = ft
        return ft


    def ifft(self, data: np.ndarray):
        """
        Inverse Fourier Transform
        """
        ft = self.fft(isign=-1)
        ft /= len(ft)
        return ft


    def power(self, data: np.ndarray=None):
        """
        One-sided power spectral density
        """

        if self._ft is None and data is None:
            raise ValueError("Argument `data` must be provided if no data "+\
                    "have been transformed yet.")
        elif self._ft is None and data is not None:
            ft = self.fft(data)
        elif data is None:
            ft = self._ft

        if self._isreal:
            ps = 2 * nps.abs(ft) ** 2
            return ps
        
        ft_pos = ft[ft >= 0]
        ft_neg = ft[ft < 0]

        return np.abs(ft_pos) ** 2 + np.abs(ft_neg) ** 2

    
    def freq(self, n: int):
        """
        Frequencies sampled by FFT
        """

        if self._isreal:
            return np.fft.rfftfreq(n, self._spacing)
        else:
            return np.fft.fftfreq(n, self._spacing)
        
        
    def period(self):
        """
        """
