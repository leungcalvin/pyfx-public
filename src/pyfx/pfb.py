"""PFB Utilities. Golden Rule: cannot import from anything in pyfx."""

import numpy as np
import scipy.linalg as la
from scipy import signal
from scipy.fft import fft, fftshift, ifft


def sinc_window(channelization):
    """Sinc window function.

    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    ntap = channelization["ntap"]
    lblock = channelization["lblock"]
    coeff_length = np.pi * ntap
    coeff_num_samples = ntap * lblock

    # Sampling locations of sinc function
    X = np.arange(
        -coeff_length / 2.0, coeff_length / 2.0, coeff_length / coeff_num_samples
    )

    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so use X/pi
    return np.sinc(X / np.pi)


def sinc_hann(channelization):
    """Hann-sinc window function.

    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    ntap = channelization["ntap"]
    lblock = channelization["lblock"]
    return sinc_window(channelization) * np.hann(ntap * lblock)


def sinc_hamming(channelization):
    """Hamming-sinc window function.

    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    ntap = channelization["ntap"]
    lblock = channelization["lblock"]
    return sinc_window(channelization) * np.hamming(ntap * lblock)


def apply_pfb(timestream, channelization, window=sinc_hamming):
    """Perform the CHIME PFB on a timestream, omitting the Nyquist frequency.

    Parameters
    ----------
    timestream : np.ndarray
        Timestream to process

    ntaps : int
        Number of taps.

    Returns
    -------
    pfb : np.ndarray[:, nfreq]
        Array of PFB frequencies.
    """
    ntap = channelization["ntap"]
    lblock = channelization["lblock"]
    nfreq = channelization["nchan"]

    # Number of blocks
    nblock = timestream.size // lblock - (ntap - 1)

    # Initialise array for spectrum
    spec = np.zeros((nblock, nfreq), dtype=np.complex128)

    # Window function
    w = window(channelization)

    # Iterate over blocks and perform the PFB
    for bi in range(nblock):
        # Cut out the correct timestream section
        ts_sec = timestream[(bi * lblock) : ((bi + ntap) * lblock)].copy()

        # Perform a real FFT (with applied window function)
        ft = np.fft.rfft(ts_sec * w)

        # Choose every n-th frequency
        spec[bi] = ft[::ntap]

    return spec


def bb_covar_kernel(delay_samples, frame_lag, freq_ind, channelization):
    """Evaluates the formula for covariance of baseband streams with a relative delay.

    Provides <b_{mk} b'_{m'k}^*>, where b' is derived from the same signal as b but delayed.
    Normalization not carefully carried through.

    Delays, frame_lag and freq_ind must be arrays can be any shape broadcastable to the
    same shape (which will be the shape of the output).

    Parameters
    ----------
    delay_samples: array
        The delay of the second stream (b') with respect to the first.
    frame_lags: array
        The difference in frame number between (m' - m)
    freq_ind: array
        The frequency index, k (which is assumed equal to k').

    Output:
    """
    lblock = channelization["lblock"]
    delay_relative_to_frame_delay = delay_samples - frame_lag * lblock
    window_auto_at_lags = pfb_window_auto_corr(
        delay_relative_to_frame_delay, channelization
    )

    phase_fac = delay_phase_factor(
        delay_samples, freq_ind, channelization=channelization
    )
    return phase_fac * window_auto_at_lags


def delay_phase_factor(delay_samples, freq_ind, channelization):
    """Calculates the frequency dependent delay_phase factor.

    Parameters
    ----------
    delay_samples: array
    freq_ind: array

    Returns
    -------
    phase_factor: array with shape of the broadcast between inputs.

    """
    nfreq = channelization["nchan"]
    lblock = channelization["lblock"]
    return np.exp(1j * 2 * np.pi * delay_samples / lblock * freq_ind)


def pfb_window_auto_corr(delay_samples, channelization):
    """The auto convolution of the pfb window, evaluated at given delays.
    Assumes PFB window is symmetric.

    """

    # Symmetry of the window function is used here... don't count on this being
    # correct for assymetric windows.
    nfreq, lblock, ntap = (
        channelization["nchan"],
        channelization["lblock"],
        channelization["ntap"],
    )
    pfb_window = sinc_hamming(channelization)
    len_window = len(pfb_window)
    window_auto_corr = signal.correlate(pfb_window, pfb_window, mode="full")
    # Roll the window auto correlation such that the 0 lag is the 0th element
    # (making later indexing easier).
    window_auto_corr = np.roll(window_auto_corr, -(len(pfb_window) - 1))
    window_auto_corr /= window_auto_corr[0]
    if False:
        window_auto_corr_lags_samples = correlation_lags(
            len(pfb_window), len(pfb_window)
        )
        window_auto_corr_lags_samples = np.roll(
            window_auto_corr_lags_samples, -(len(pfb_window) - 1)
        )
        plt.figure()
        plt.plot(
            window_auto_corr_lags_samples, window_auto_corr.real, ls="", marker="."
        )
    delay_samples = np.round(delay_samples).astype(int)
    window_auto_at_lags = window_auto_corr[delay_samples % len(window_auto_corr)]
    window_auto_at_lags[abs(delay_samples) >= len_window] = 0
    return window_auto_at_lags


def chime_pfb(timestream):
    return apply_pfb(timestream, config.CHIME_PFB)[:, :-1:]


def calculate_inv_noise_covar_kernel(channelization):
    """Calcualte the inverse noise covariance kernel, given channelization properties.

    Resulting kernel inverse covariance weights when convolved through the
    data. This accounts for sample-to-sample correlations in the data caused by
    the PFB.

    """
    nfreq = channelization["nchan"]
    # PFB introduces correlations spanning these frame lags.
    noise_covar_len = 2 * channelization["ntap"] - 1
    noise_covar_lags = np.arange(noise_covar_len) - noise_covar_len // 2
    # Calculate the covariance at these lags.
    noise_covar_kernel = bb_covar_kernel(0, noise_covar_lags, 0, channelization)
    # here, call bb_covar_kernel
    # ... frame_lag = 0 because most of the total covariance comes from the system temperature autocorrelating at zero, and
    # ... freq_ind = 0 to omit the phase factor from the covariance.

    # And the inverse covariance.
    inv_noise_covar_kernel = ifft(
        1.0 / fft(noise_covar_kernel)
    ).real  # Kiyo math, don't understand
    # Fix off-by-1 error in the translational part of the kernel which was not properly centered on
    # 0.
    inv_noise_covar_kernel = np.roll(inv_noise_covar_kernel, -1)
    # Debug plot of (inverse) noise covariance kernels.
    if False:
        plt.figure()
        plt.plot(noise_covar_lags, noise_covar_kernel, ".")
        plt.plot(noise_covar_lags, inv_noise_covar_kernel, ".")
        plt.xlabel("lag (frames)")
        plt.ylabel("kernel")
    return inv_noise_covar_kernel, noise_covar_lags
