from typing import Optional, Tuple

import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
import time


from baseband_analysis.core.bbdata import BBData
from difxcalc_wrapper.io import IMReader

from pyfx.core_math import fft_corr
from pyfx.core_math import max_lag_slice

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)

def autocorr_core(DM, bbdata_A, T_A, Window, R, max_lag=None,n_pol=2):
    """Correlates and downselects over lag (potentially more lags at shorter integration times
    DM - the DM with which we de-smear the data before the final gating. for steady sources, set dispersion measure to 0.
    bbdata_A - baseband data
    T_A[i,j] - start times at ith frequency, for jth time chunk, for telescope A
    Window[i,j] - length of time chunk window (us)
    R[i,j] - fraction of time chunk (defines pulse window). Variable name should be more descriptive
    max_lag - maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data)
    n_pol - number of polarizations in data
    """
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(T_A,axis=-1)
    n_pointings=bbdata_A["tiedbeam_baseband"].shape[1] // 2 ## SA: basing this off of how the data is arranged now, may want to change

    if (type(R) is np.ndarray)==False:
        R=np.full(Window.shape, R)

    if max_lag is None:
        max_lag = np.max(
            Window
        )  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))
    auto_vis = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)

    ## will compress this horrific number of for loops later
    for pointing in range(n_pointings):
        for iifreq in range(n_freq):
            for iipol in range(n_pol):
                for jjpol in range(n_pol):
                    for iitime in range(n_scan):
                        t_ij = T_A[iifreq, iitime]
                        w_ij=Window[iifreq,iitime]
                        r_ij=R[iifreq,iitime]

                        start = int((w_ij - w_ij*r_ij) // 2)
                        stop = int((w_ij + w_ij*r_ij) // 2)

                        _vis = fft_corr(
                            bbdata_A['tiedbeam_baseband'][
                                iifreq,
                                iipol,
                                start : stop,
                            ],
                            bbdata_A['tiedbeam_baseband'][
                                iifreq,
                                jjpol,
                                start : stop,
                            ])
                        auto_vis[iifreq, pointing, iipol, jjpol, iitime,:] = np.concatenate((_vis[:max_lag+1],_vis[-max_lag:]))


    return auto_vis


def crosscorr_core(
    bbdata_A: BBData,
    bbdata_B: BBData,
    T_A: np.ndarray,
    Window: np.ndarray,
    R: np.ndarray,
    calc_results: IMReader,
    DM: int,
    index_A: int=0,
    index_B: int=1,
    sample_rate: float=2.56,
    max_lag: Optional[int]=None,
    n_pol: int=2
):
    """
    inputs:
    bbdata_A - telescope A data (sliced into a frequency chunk)
    bbdata_B - telescope B data (sliced into a frequency chunk)
    T_A[i,j] - starting frames at ith frequency, for jth time chunk, for telescope A
    Window[i,j] - np.array (nfreq,nscan) holding length of scan window in frames
    R[i,j] - fraction of time chunk (defines pulse window). For steady sources, R=1 ("on" window = full window)
    DM - the DM with which we de-smear the data before the final gating. for steady sources, set dispersion measure to 0.
    calc_results - difxcalc object containing
    index_A - where telescope A corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    index_B - where telescope B corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    sample_rate - rate at which data is sampled in microseconds
    max_lag - maximum (absolute value) lag (in frames) for correlations (useful for very long time series data)

    Outputs:
    cross - array of autocorrelations and cross correlations with shape (pointing,freq, timechunk, pol, pol, delay)

    """
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(T_A, axis=-1) # which chunk (1 for small time scales)

    # bbdata_A["tiedbeam_baseband"] - (freq, pointing, time)
    n_pointings=bbdata_A["tiedbeam_baseband"].shape[1] // 2 ## SA: basing this off of how the data is arranged now, may want to change

    # initialize output autocorrelations and cross correlations
    if max_lag is None:
        max_lag = 100  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.

    # output - freq, pointing, pol1, pol2, scan, window of cross correlation we want
    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))

    cross = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)

    total_retarded_baseline_delay_time = 0
    for pointing in range(n_pointings):
        for iifreq in range(n_freq):
            ### require user to have "well-ordered" bbdata in frequency (iifreqA=iifreqB)
            f0 = bbdata_B.freq[iifreq] ##frequency centers in MHz
            for jjscan in range(n_scan):
                w_ij = Window[iifreq, jjscan]
                r_ij = R[iifreq, jjscan]
                T_A_index = T_A[iifreq, jjscan]
                t0_a = bbdata_A["time0"]["ctime"][iifreq] + T_A_index*(sample_rate*1e-6)
                ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0.
                ## get array of geometric delay over the scan (i.e .as a function of time)

                #  
                start_time = Time(
                    t0_a,
                    val2=bbdata_A["time0"]["ctime_offset"][iifreq],
                    format="unix",
                    precision=9,
                )


                query_times = start_time + sample_rate*1e-6 * un.s * (T_A_index+np.arange(w_ij)) #also calculating 100 frames ahead as cushion
                # using telescope A times as reference time
                ### should probably just call this once out of the for loop using np.flatten but will fix later
                start = time.time()
                geodelay: float = calc_results.retarded_baseline_delay(
                    ant1=index_A, ant2=index_B, time=query_times, src=pointing,delay_sign=0,self_consistent=False
                )
                end = time.time()
                total_retarded_baseline_delay_time += end-start

                ### Fringestopping B -> A
                scan_a, scan_b_fs = get_aligned_scans(
                    bbdata_A, bbdata_B,T_A_index, w_ij, geodelay, freq_id=iifreq,sample_rate=sample_rate
                )

                #######################################################
                ######### intrachannel de-dispersion Time. ############
                scan_a_cd = intrachannel_dedisp(scan_a, DM, f0=f0)
                scan_b_fs_cd = intrachannel_dedisp(scan_b_fs, DM, f0=f0)

                #######################################################
                ### Now that the pulses are centered at zero, calculate
                ### the start and stop time indices for on-signal ######
                start = int((w_ij - w_ij*r_ij) // 2)
                stop = int((w_ij + w_ij*r_ij) // 2)
                #######################################################
                ########## cross-correlate the on-signal ##############
                for pol_0 in range(n_pol):
                    for pol_1 in range(n_pol):
                        _vis = fft_corr(
                            scan_a_cd[pol_0, start:stop],
                            scan_b_fs_cd[pol_1, start:stop])
                        cross[iifreq, pointing, pol_0, pol_1, jjscan,:] = np.concatenate((_vis[:max_lag+1],_vis[-max_lag:]))
    print(f"total time running retarded_baseline_delay: {total_retarded_baseline_delay_time}")
    return cross
    


def intrachannel_dedisp(
    data: np.ndarray,
    DM: float,
    f0: np.ndarray, 
    sample_rate: float = 2.56
):
    """Intrachannel dedispersion: brings data to center of channel.

    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook, but ONLY the last term (proportional to f^2), not the other two terms (independent of f and linearly proportional to f respectively).

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    sample_rate : sampling rate of data in microseconds

    TODO: use numba or GPU for this!"""
    #n = data.shape[-1]
    #f = np.fft.fftfreq(n) * sample_rate
    #f0=np.array(f0)
    #for iifreq, _f0 in enumerate(f0):
    #    transfer_func = np.exp(
    #        -2j * np.pi * K_DM * DM * f**2 / _f0**2 / (f + _f0)
    #    )  # double check this minus sign -- might be a + instead in CHIME data.
    #    data[iifreq, ...] = ifft(fft(data[iifreq, ...], axis=-1) * transfer_func)
    #return data

    n = data.shape[-1]
    f = np.fft.fftfreq(n) * sample_rate
    transfer_func = np.exp(-2j * np.pi * K_DM * DM * f**2 / f0**2 / (f + f0))  # double check this minus sign -- might be a + instead in CHIME data.
    data = np.fft.ifft(np.fft.fft(data, axis=-1) * transfer_func)
    return data


def frac_samp_shift(
    data: np.ndarray,
    f0: float,
    sub_frame_tau: Optional[np.ndarray] = None,
    sample_rate: float = 2.56,
    freq_id: Optional[int] = None,
    complex_conjugate_convention: int = -1,
    intra_channel_sign: int = 1
):
    """Fractional sample correction: coherently shifts data within a channel.

    data : np.ndarray of shape (ntime)

    f0 : frequency channel center.

    sample_rate : sampling rate of data in microseconds

    sub_frame_tau: np.array of shape (ntime), sub-frame delay in us

    complex_conjugate_convention: a sign to account for the fact that the data may be complex conjugated

    intra_channel_sign: a sign to account for a reflection of frequencies about zero (e.g. in iq/baseband data)

    Applies a fractional phase shift of the form exp(2j*pi*f*sub_frame_tau) to the data.

    ## need to rethink looping over frequency in the main function; this should take in an array of freqs
    """
    n = data.shape[-1]
    f = np.fft.fftfreq(n, sample_rate)
    transfer_func = np.exp(intra_channel_sign*2j * np.pi * f * np.median(sub_frame_tau,axis=-1))  # apply dphi/dfreq
    data2 = np.fft.ifft(
        np.fft.fft(data,axis=-1) * transfer_func
    ) * np.exp(
        complex_conjugate_convention*2j * np.pi * f0 * sub_frame_tau
    )  # apply phi
    return data2


def get_aligned_scans(
    bbdata_A: BBData,
    bbdata_B: BBData,
    T_A_index, #ACTUALLY AN INT??
    wij: int,
    tau: np.array,
    freq_id: int,
    sample_rate = 2.56
) -> Tuple[np.array, np.array]:
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.

    bbdata_A : BBData
        A BBData object, with arbitrary frequency coverage.

    bbdata_B : BBData
        A BBData object, with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    T_A_index : np.array of shape (1024, n_time)
        An array of indices corresponding to the start frames for telescope A
        TODO(shion) This is an int, not an array

    w_ij : int
        A particular value w_ij for this baseline. Should be an integer, and brownie points for a good FFT length.

    tau : np.array (n_frame) of dtype np.float
        A delay in microseconds to apply to BBData_b, corresponding to the geometric delay.
        The first index is the delay evaluated at time t_ij_a

    freq_id : int

    sample_rate : float

    Returns
    -------
    aligned_a : np.array
        A dual-pol scan of shape (2, w_ij)

    aligned_b : np.array
        A dual-pol scan of shape (2, w_ij)

    # newstart: int
    #     Number of frames by which we need to shift T_A_ij in order to ensure T_A_ij+geodelay is contained within bbdata_B. Note that in the event that geodelay is positive, newstart will always be 0 (assuming the user has chosen T_A_ij such that the unix time is in both datasets)

    Super technical note on floor vs round: it doesn't matter AS LONG AS you do a sub-sample rotation (or better yet, a frac samp correction)! Suppose your total delay is 10.6 frames.
    - You can round to 11 frames. You should keep track that you rounded to 11, and then do frac samp -0.4.
    - You can floor to 10 frames, you should keep track that you floored to 10, and then do frac samp +0.6.

    Answer should be the same either way -- as long as you do the frac samp correction!
    After doing the integer part (shift by either 10 or 11 frames), we need to apply a phase rotation. Note that exp(2j*np.pi*channel_center * -0.4/(2.56us) = exp(2j*np.pi*channel_center * +0.6/(2.56us), for the exact frequency corresponding to channel center, but not for any of the other frequencies that do not satisfy f = 800 - (N * 0.390625 MHz) for integers N -- this is the narrowband approximation. We experience some de-correlation near the band edges, which is why we use fractional sample correction in this code.
    """

    time_we_want_at_b = tau[0] #us

    # bbdata_A["tiedbeam_baseband"] - (freq, pointing, time)
    aligned_a = bbdata_A['tiedbeam_baseband'][freq_id,...,T_A_index:T_A_index + wij]
    aligned_b = np.zeros(bbdata_B['tiedbeam_baseband'][freq_id].shape,dtype=bbdata_B['tiedbeam_baseband'].dtype) #initialize aligned B array
    ## calculate the additional offset between A and B in the event that the (samples points of) A and B are misaligned in absolute time by < 1 frame
    ## i.e. to correctly fringestop, we must also account for a case such as:
    ## A:    |----|----|----|----|----| ##
    ## B: |----|----|----|----|----|    ##
    t_a= Time(
        bbdata_A["time0"]["ctime"][freq_id],
        val2=bbdata_A["time0"]["ctime_offset"][freq_id],
        format="unix",
        precision=9)
    t_b= Time(
        bbdata_B["time0"]["ctime"][freq_id],
        val2=bbdata_B["time0"]["ctime_offset"][freq_id],
        format="unix",
        precision=9)

    delta_A_B=(t_b-t_a).to_value('sec') ## offset in time which already exists between telescope A and B
    int_delay=int(np.round((time_we_want_at_b*1e-6 - delta_A_B) / (sample_rate*1e-6)))

    start_index_we_want_at_b = T_A_index+int_delay #frame number closest to start time
    start_index_we_have_at_b = np.max([start_index_we_want_at_b, 0])# account for case where T_A_index+geodelay < 0 (i.e. signal arrives at telescope B before start of data acquision)
    pad_index_b=start_index_we_have_at_b-start_index_we_want_at_b #if index_we_have_at_b is negative, this will be the amount we need to cushion our output data by

    stop_index_we_have_at_b = np.min([start_index_we_have_at_b+wij,bbdata_B.ntime-start_index_we_have_at_b])
    stop_index_we_have_at_b=np.min([stop_index_we_have_at_b,bbdata_B.ntime-pad_index_b])

    new_wij=int(stop_index_we_have_at_b-start_index_we_have_at_b)
    correction_factor = wij / new_wij # if you are missing half the data, multiply by 2.
    if correction_factor > 2:
         # warn the user that the boundary conditions are sketch if we are missing e.g. more than half the data.
         print("warning: based on specified start time and scan length, over half the data is missing from telescope XX.")

    # TODO(shion) start_index_we_have_at_b:start_index_we_have_at_b??
    aligned_b[...,pad_index_b:pad_index_b+new_wij] = \
        bbdata_B['tiedbeam_baseband'][freq_id,...,start_index_we_have_at_b:start_index_we_have_at_b+new_wij] * correction_factor

    # multiply by the correction factor to ensure that a steady source, when correlated,
    # has the correct flux corresponding to the desired w_ij, even when we run out of data.
    # TODO(shion) how is this multiplying?
    aligned_b=aligned_b[...,:wij]

    # current start time at B (relative to start time of A )= delta_A_B+start_index_we_want_at_b*sample_rate*1e-6-(T_A_index*sample_rate*1e-6)
    # = delta_A_B+int_delay
    ### calculate remaining sub-frame delay
    time_we_have_at_b = (delta_A_B+int_delay*sample_rate*1e-6) #s
    sub_frame_tau=(tau[:wij] - time_we_have_at_b*1e6) #sub-frame delay at start time in mircoseconds

    aligned_b = frac_samp_shift(
        aligned_b,
        f0=bbdata_B.index_map["freq"]["centre"][freq_id],
        sub_frame_tau=sub_frame_tau,
        sample_rate=sample_rate,
        freq_id=freq_id
    )
    return aligned_a, aligned_b