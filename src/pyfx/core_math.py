"""A module which holds bindings to mathematical operations. This module is meant to be used on *baseband* data"""
import numpy as np
from scipy.fft import fft, ifft, next_fast_len,fftfreq
import torch
import torch.fft as torch_fft
import time

# note - scipy.fft is faster than numba.fft
def fft_corr_gpu(
    w1: torch.Tensor, 
    w2: torch.Tensor, 
    axis=-1
):
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).
    out : torch.Tensor"""
    assert axis==-1, "fft in pytorch is only supported along last axis of data"
    x=torch_fft.fft(w1)
    del(w1)
    y=torch.conj(torch_fft.fft(w2))
    del w2
    return torch_fft.ifft(x*y)  

# S.E.A.: note - scipy.fft is faster than np.fft. 
# Scipy.fft is already optimized to the point where that implementing this in cython won't really speed things up; you need gpus (see core_math_torch.py)
def fft_corr(
    w1: np.ndarray, 
    w2: np.ndarray, 
    axis=-1
):
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).
    
    Inputs:
    -------
    w1 : np.ndarray
    w2 : np.ndarray

    Outputs:
    -------
    out : np.ndarray"""
    #Correlation between w1 and w2: the output will have lag zero at index 0.
    #out[1:n//2] contains positive lags, out[n//2+1] contains negative lags"
    return ifft(fft(w1, axis=axis) * fft(w2, axis=axis).conj(), axis=axis)

def max_lag_slice(
    data: np.ndarray, 
    max_lag: int, 
    lag_axis: int=-1
):
    """Downselects an array out to some maximum lag.
    Inputs:
    -------
    data : np.ndarray
        Some array of shape, with lag as one of its axes.

    Outputs:
    -------
    out : np.ndarray
        An array of the same shape as array, except the lag axis has been downselected to (2 * max_lag + 1).
    """
    out = np.zeros(shape=shape, dtype=data.dtype)
    within_n_lags=np.full(np.size(data,axis=lag_axis),False)
    within_n_lags[:max_lag+1]=True
    within_n_lags[-max_lag:]=True
    return np.compress(within_n_lags, a=data, axis=lag_axis, out=out)


def xy_to_circ(bb):
    assert bb.shape[1] == 2, "Baseband is not dual-polarization!"
    if bb.ndim != 3:
        UserWarning("Data shape is {bb.shape}! Expected 3 dimensions.")
    out = np.zeros_like(bb)
    out[:, 0] = bb[:, 0] + 1j * bb[:, 1]
    out[:, 1] = bb[:, 0] - 1j * bb[:, 1]
    return out
