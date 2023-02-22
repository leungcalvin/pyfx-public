"""A module which holds bindings to mathematical operations. Here we only import numpy and scipy."""
import numpy as np
from scipy import fft


def fft_corr(w1, w2, axis=-1):
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).

    w1 : np.ndarray

    w2 : np.ndarray

    out : np.ndarray
        The output will have lag zero at index 0."""
    return ifft(fft(w1, axis=axis) * fft(w2, axis=axis).conj(), axis=axis)


def max_lag_slice(array, max_lag, lag_axis=-1):
    """Downselects an array out to some maximum lag.

    array : np.ndarray
        Some array of shape, with lag as one of its axes.

    out : np.ndarray
        An array of the same shape as array, except the lag axis has been downselected to (2 * max_lag + 1).
    """
    shape = list(array.shape)
    shape[lag_axis] = 2 * max_lag + 1
    out = np.zeros(shape=shape, dtype=array.dtype)
    within_n_lags = (
        np.abs(fft.fftfreq(array.shape[lag_axis]) * array.shape[lag_axis]) < max_lag
    )
    return np.compress(within_n_lags, a=array, axis=lag_axis, out=out)


def xy_to_circ(bb):
    assert bb.shape[1] == 2, "Baseband is not dual-polarization!"
    if bb.ndim != 3:
        UserWarning("Data shape is {bb.shape}! Expected 3 dimensions.")
    out = np.zeros_like(bb)
    out[:, 0] = bb[:, 0] + 1j * bb[:, 1]
    out[:, 1] = bb[:, 0] - 1j * bb[:, 1]
    return out
