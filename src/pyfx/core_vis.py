"""A module which performs simple functions on visibilities. This module is meant to be used on *visibility* data"""

import time
from typing import List, Tuple

import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft, next_fast_len
from scipy.stats import median_abs_deviation


def extract_frame_delay(
    vis: np.ndarray, sample_rate: float = 2.56, lag_axis: int = -1, n_pol: int = 2
) -> List[int]:
    """Extract integer frame delay from visibilities that maximizes snr

    Inputs:
    ----------
    vis - np.array of visibilities of shape (nfreq, npol, npol, nlag)

    Outputs:
    -------
    tau : The delay (in units of microseconds) maximizing the SNR of cros-correlation.
    rhosf : The grid of cross-correlation values.

    """
    shifted_vis = fftshift(vis, axes=lag_axis)
    resid_frame_lag = fftshift(np.arange(vis.shape[lag_axis]))
    nframes = np.size(shifted_vis, axis=lag_axis)
    resid_frame_lag[resid_frame_lag > nframes // 2] -= nframes
    norm_vis = np.abs(shifted_vis)
    norm_vis[~np.isfinite(norm_vis)] = np.median(norm_vis)
    xcorr_incoherent_avg = np.nanmean(norm_vis, axis=0)  # average over frequency
    peaklags = []
    for pol in range(n_pol):
        peaklags.append(resid_frame_lag[np.argmax(xcorr_incoherent_avg[pol, pol])])
    return peaklags


def extract_subframe_delay(
    vis: np.ndarray, sample_rate: float = 2.56, peak_lag: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sub-frame delay from visibilities that maximizes snr

    Inputs:
    ----------
    vis :
        np.array of visibilities of shape (nfreq, npol, npol, nlag)

    sample_rate :
        rate at which data is sampled in microseconds

    Outputs:
    -------
    tau : The delay (in units of microseconds) maximizing the SNR of cros-correlation.
    rhosf : The grid of cross-correlation values.

    """
    x = fftfreq(n=2**15) * sample_rate  # microseconds
    vis = vis[:, :, :, peak_lag]
    vis[np.isnan(vis)] = 0
    rhosf = np.abs(ifft(vis, n=2**15, axis=0))
    rhosf /= median_abs_deviation(rhosf)  # normalize
    snr = np.max(np.abs(rhosf), axis=0)
    tau = np.take(x, np.argmax(np.abs(rhosf), axis=0), axis=0)
    return tau, snr
