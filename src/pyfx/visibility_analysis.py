"""A module which performs simple functions on visibilities. This module is meant to be used on *visibility* data"""
import numpy as np
from scipy.fft import fft, ifft, next_fast_len,fftfreq
import torch
import torch.fft as torch_fft
import time

def extract_subframe_delay(
    data: np.ndarray,
    sample_rate: float=2.56,
    ):
    """Extract sub-frame delay from visibilities that maximizes snr

    Inputs:
    ----------
    data - np.array of complex values
        Complex data, regularly sampled, that has a linear phase dependence.

    Outputs:
    -------
    tau : The delay (in units of microseconds) maximizing the SNR of cros-correlation.
    rhosf : The grid of cross-correlation values.

    """
    x = fftfreq(n=2**15) * 2.56
    data[np.isnan(data)] = 0
    rhosf = np.abs(ifft(data, n=2**15, axis=axis))
    rhosf /= median_abs_deviation(rhosf)
    snr=np.max(np.abs(rhosf))
    tau = np.take(
        x, np.argmax(rhosf, axis=axis), axis=axis)
    return tau, rhosf, snr
