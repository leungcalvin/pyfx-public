import numpy as np
from scipy.fft import fft, ifft, next_fast_len,fftfreq,fftshift
from scipy import signal
from pyfx import config,pfb

# S.E.A.: note - scipy.fft is faster than np.fft. 
# Scipy.fft is already optimized to the point where that implementing this in cython won't really speed things up; you need gpus (see core_math_torch.py)

CHANNELIZATION = config.CHANNELIZATION # grab config from config.py

"""Precompute PFB kernels.
INV_NOISE_COVAR_KERNEL is an array of shape (2 * ntap - 1)
SIG_SEARCH_KERNEL is an array of shape (n_search_lags, 2 * ntap - 1)
"""

IVW_KERNEL, IVW_LAGS = pfb.calculate_inv_noise_covar_kernel(CHANNELIZATION)
SIG_SEARCH_KERNEL = np.zeros((len(CHANNELIZATION['search_lags']),
                          2 * CHANNELIZATION['ntap'] - 1)
                        )
for ii, subframe_delay in enumerate(CHANNELIZATION['search_lags']):
    print(f'Precomputing delay-search PFB coefficients: {ii} of {len(CHANNELIZATION["search_lags"])}')
    SIG_SEARCH_KERNEL[ii,:] = pfb.pfb_window_auto_corr(IVW_LAGS * CHANNELIZATION['lblock'] - subframe_delay, 
    CHANNELIZATION)
CHANNELIZATION['IVW_KERNEL'] = IVW_KERNEL
CHANNELIZATION['SIG_SEARCH_KERNEL'] = SIG_SEARCH_KERNEL

def fft_corr(
    w1: np.ndarray, 
    w2: np.ndarray, 
    axis=-1
 ) -> np.ndarray:
    
    """Correlates but vectorizes over all axes except the correlation axis (-1 by default).
    
    Inputs:
    -------
    w1 : np.ndarray
    w2 : np.ndarray

    Outputs:
    -------
    out : np.ndarray
        w1 cross cross correlated with w2
    """

    #Correlation between w1 and w2: the output will have lag zero at index 0.
    #out[1:n//2] contains positive lags, out[n//2+1] contains negative lags"
    return ifft(fft(w1, axis=axis) * fft(w2, axis=axis).conj(), axis=axis)

def basic_correlator(w1, w2, max_lag=None,channelization = CHANNELIZATION, full_output = False):
    """Basic correlator.
    """
    if max_lag is None:
        max_lag=channelization['nlags']
    cross_corr_func = fft_corr(w1, w2)
    if full_output: 
        frame_lags = fftfreq(cross_corr_func.shape[-1]) * cross_corr_func.shape[-1]
        return cross_corr_func, frame_lags
    return max_lag_slice(cross_corr_func,max_lag = max_lag,lag_axis = -1)


def inverse_variance_weight_correlator(w1, w2, max_lag=None,channelization = CHANNELIZATION,full_output = False):
    """Inverse noise weighting correlator.

    This which accounts for the sample-to-sample correlations *within* a
    baseband dataset, but not the correlations between datasets. This should be
    optimal for the integer delay case, but I don't think this should be a
    large improvement over the basic estimator.

    """
    # Noise weight the data by convolving with the inverse covariance kernel.
    if max_lag is None:
        max_lag=channelization['nlags']
    w1_ivw = convolve_bb_time_kernel(w1, kernel = channelization['IVW_KERNEL'])
    cross_corr_func = fft_corr(w1_ivw, w2)

    if full_output: 
        frame_lags = fftfreq(cross_corr_func.shape[-1]) * cross_corr_func.shape[-1]
        return cross_corr_func, frame_lags
    return max_lag_slice(cross_corr_func,max_lag = max_lag,lag_axis = -1)

def signal_to_noise_weight_correlator(w1, w2, signal_kernel, max_lag=None,channelization = CHANNELIZATION, full_output = False):
    """The optimal delay estimator for a known delay.

    Not practical in the real world, since it requires knowledge of the delay
    before it is estimated, but provides a limiting case for simulations.

    Parameters
    ----------
    signal_kernel : array of length (ntap * 2 + 1,)
        Must provide signal_kernel coefficients along with data arrays.
    """
    if max_lag is None:
        max_lag=channelization['nlags']
    nsamp_frame = channelization['lblock']
    w1_ivw = convolve_bb_time_kernel(w1, kernel = channelization['IVW_KERNEL'])
    w2_ivw = convolve_bb_time_kernel(w2, kernel = channelization['IVW_KERNEL'])
    w1_sivw = convolve_bb_time_kernel(w1_ivw, kernel = signal_kernel)
    cross_corr_func = fft_corr(w1_sivw, w2_ivw)
    
    if full_output: 
        frame_lags = fftfreq(cross_corr_func.shape[-1]) * cross_corr_func.shape[-1]
        return cross_corr_func, frame_lags
    return max_lag_slice(cross_corr_func,max_lag = max_lag,lag_axis = -1)

def subframe_signal_to_noise_search_correlator(data_bb_1, data_bb_2,
        channelization = CHANNELIZATION,full_output = False):
    """Perform optimal search for a set of given trial subframe lags.

    Use highest SNR to determine best search and return estimate from that
    search.

    """
    nchan = channelization['nchan']
    nsamp_frame = channelization['lblock']
    n_search_lags = len(channelization['search_lags'])
    # Inverse covariance weight the baseband data.
    w1_ivw = convolve_bb_time_kernel(data_bb_1, kernel = channelization['IVW_KERNEL'])
    w2_ivw = convolve_bb_time_kernel(data_bb_2, kernel = channelization['IVW_KERNEL'])
    # Loop over trial subframe delays and correlate the data optimized for each.
    
    for ii,subframe_delay in enumerate(channelization['search_lags']):
        print(f'Searching subframe_delay {ii} of {n_search_lags}')
        # The trial delay and signal template prediction at that delay.
        # Signal weight the data and correlate.
        w1_sivw = convolve_bb_time_kernel(w1_ivw, kernel = channelization['SIG_SEARCH_KERNEL'][ii,:])
        this_cross_corr_func = fft_corr(w1_sivw,w2_ivw)
        this_frame_lags = subframe_delay / channelization['lblock'] + fftfreq(w1_sivw.shape[-1]) * w1_sivw.shape[-1]

        # Recenter the frame lags to include the trial delay; perform the
        # corresponding compensation of the correlation function phase.
        # CL: Skip the recentering and phasing in PyFX? No, we want all quasi-integer lags to correspond to the same sub-frame delay.
        # this_frame_lags = this_frame_lags + ii / subframe_granularity
        assert w1_sivw.shape[0] == channelization['nchan'], "Wrong number of channels; search correlator requires 1024 channels at present."


        if ii == 0:
            # Initialize global output arrays
            nframe_lags = len(this_frame_lags)
            frame_lags = np.zeros((n_search_lags, nframe_lags),
                    this_frame_lags.dtype)
            cross_corr_func = np.zeros((this_cross_corr_func.shape[0], n_search_lags, this_cross_corr_func.shape[-1]),
                    this_cross_corr_func.dtype)
        # Store the output of this trial subframe lag.
        frame_lags[ii, :] = this_frame_lags
        cross_corr_func[:, ii, :] = this_cross_corr_func

    # Now we need to re-order everything to match the original convention.
    # We fftshift, then transpose to sort everything as a function of increasing quasi-frame delay
    frame_lags = fftshift(frame_lags,axes = (0,1),).swapaxes(0,1).reshape((nframe_lags * n_search_lags))
    cross_corr_func = fftshift(cross_corr_func,axes = (1,2)).swapaxes(1,2).reshape((nchan,nframe_lags * n_search_lags)) 
    
    # Finally we fftshift again on the flattened array to obey zero lag at index=0 convention.
    frame_lags = np.roll(fftshift(frame_lags,axes = (-1)), -n_search_lags // 2, axis = -1)
    cross_corr_func = np.roll(fftshift(cross_corr_func,axes = -1), -n_search_lags // 2, axis = -1)
    if full_output: return cross_corr_func, frame_lags
    return max_lag_slice(cross_corr_func,max_lag = channelization['nlags'],lag_axis = -1)

def convolve_bb_time_kernel(data_bb, kernel):
    """Convolve baseband data by a provided kernel over the time (frame) axis.

    Parameters:
    data_bb: complex array with shape (nfreq, nframe)
    kernal: kernel with which to convolve

    This just loops over the frequency axis and calls `np.convolve`. The convolution mode is "same",
    meaning the first and last `len(kernel)//2` are subject to periodic boundary condition effects. 
    However, for symmetric kernels it does not shift anything in time.
    """

    out = np.empty_like(data_bb)
    for iif in range(data_bb.shape[0]):
        # Use mode="same" to not change the length of the data. This corrupts
        # len(kernel)//2 = 3 frames on either end, but these get killed by the
        # window at correlation time anyway.
        out[iif] = np.convolve(data_bb[iif], kernel, mode="same")
    return out

def max_lag_slice(
    data: np.ndarray, 
    max_lag: int, 
    lag_axis: int=-1
 ) -> np.ndarray:
    
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
    shape = list(data.shape)
    shape[lag_axis] = 2 * max_lag + 1
    out = np.zeros(shape=shape, dtype=data.dtype)
    within_n_lags=np.full(np.size(data,axis=lag_axis),False)
    within_n_lags[:max_lag+1]=True
    within_n_lags[-max_lag:]=True
    return np.compress(within_n_lags, a=data, axis=lag_axis, out=out)
