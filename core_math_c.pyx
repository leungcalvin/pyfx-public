"""A module which holds bindings to mathematical operations. Here we only import numpy and scipy."""
import numpy as np
from scipy.fft import fft, ifft, next_fast_len,fftfreq
import numba
from numba import njit

# @njit numba no support fft, so we _could_ try @jit, but this will work with
# python objects and would likely be slow
# note - scipy.fft is faster than numba.fft
def fft_corr(w1: np.ndarray, w2: np.ndarray, axis=-1):
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).

    w1 : np.ndarray

    w2 : np.ndarray

    out : np.ndarray
        Correlation between w1 and w2: the output will have lag zero at index 0.
        out[1:n//2] contains positive lags, out[n//2+1] contains negative lags"""

    return ifft(fft(w1, axis=axis) * fft(w2, axis=axis).conj(), axis=axis)


