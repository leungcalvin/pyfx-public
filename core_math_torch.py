"""A module which holds bindings to mathematical operations. Here we only import numpy and scipy."""
import numpy as np
from scipy.fft import fft, ifft, next_fast_len,fftfreq
import numba
import torch
import torch.fft as torch_fft
import time

# @njit numba no support fft, so we _could_ try @jit, but this will work with 
# python objects and would likely be slow
# note - scipy.fft is faster than numba.fft
def fft_corr_gpu(w1: torch.Tensor, w2: torch.Tensor, axis=-1):
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).
    out : torch.Tensor
        Correlation between w1 and w2: the output will have lag zero at index 0.
        out[1:n//2] contains positive lags, out[n//2+1] contains negative lags"""
    assert axis==-1, "fft in pytorch is only supported along last axis of data"
    x=torch_fft.fft(w1)
    del(w1)
    y=torch.conj(torch_fft.fft(w2))
    del w2
    return torch_fft.ifft(x*y)  


