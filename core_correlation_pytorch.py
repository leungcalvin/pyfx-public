import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
import time

from pyfx.core_math_torch import fft_corr_gpu
#from pyfx.core_math_c import fft_corr
from pyfx.core_math import max_lag_slice

#enable type hints for static tools
from baseband_analysis.core.bbdata import BBData
from difxcalc_wrapper.io import IMReader
from typing import Optional, Tuple, Union
import torch

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def autocorr_core_gpu(
    DM: float,
    bbdata_A: BBData,
    t_a: np.ndarray,
    window: Union[np.ndarray, int],
    R: Union[np.ndarray, float],
    max_lag: int=None,
    n_pol: int=2):
    ## assumes window is constant and R varies vs time
    ## this is not yet properly vectorized for variable t_a
    """Correlates and downselects over lag (potentially more lags at shorter integration times
    DM - the DM with which we de-smear the data before the final gating. for steady sources, set dispersion measure to 0.
    bbdata_A - baseband data
    t_a[i,j] - start times at ith frequency, for jth time chunk, for telescope A
    window[j] - integer or np.array of size (nscan) holding length of time chunk window (us)
    R[i,j] - integer or np.array of size (nfreq,nscan). Fraction of time chunk (defines pulse window). Variable name should be more descriptive
    max_lag - maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data)
    n_pol - number of polarizations in data
    """
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(t_a, axis=-1)
    # SA: basing this off of how the data is arranged now, may want to change
    n_pointings = bbdata_A["tiedbeam_baseband"].shape[1] // 2

    if max_lag is None:
        max_lag = np.max(
            window
        )  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))
    auto_vis = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)

    # will compress this horrific number of for loops later
    for pointing in range(n_pointings):
        for iipol in range(n_pol):
            for jjpol in range(n_pol):
                for jjscan in range(n_scan):
                    if type(window)==int:
                        window_jjscan=window
                    else:
                        window_jjscan=window[jjscan]

                    t_a_indices = t_a[:, jjscan]  # array of length 1024

                    if type(R)==int: #should be 1 for steady sources
                        r_jjscan=R
                    elif len(np.unique(R[:,jjscan]))==1:
                        r_jjscan=R[0,jjscan]

                    else: # "on" window varies as a function of frequency (e.g. pulsar)
                        r_jjscan=R[:,jjscan] #np array of size (nfreq)
                        if len(np.unique(r_jjscan))==1:
                            r_jjscan=r_jjscan[0]

                    if (type(r_jjscan)==float or type(r_jjscan)==int) and len(np.unique(t_a_indices))==1:
                        r_ij=r_jjscan
                        start = int((window_jjscan - window_jjscan*r_ij) // 2)+t_a_indices[0]
                        stop = int((window_jjscan + window_jjscan*r_ij) // 2)+t_a_indices[0]

                        _vis = fft_corr(
                            bbdata_A['tiedbeam_baseband'][
                                :,
                                iipol,
                                start: stop,
                            ],
                            bbdata_A['tiedbeam_baseband'][
                                :,
                                jjpol,
                                start: stop,
                            ])
                        auto_vis[:, pointing, iipol, jjpol, jjscan, :] = np.concatenate((_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
                    else:
                        for iifreq,r_ij in enumerate(r_jjscan):
                            start = int((window_jjscan - window_jjscan*r_ij) // 2)+t_a_indices[iifreq]
                            stop = int((window_jjscan + window_jjscan*r_ij) // 2)+t_a_indices[iifreq]

                            _vis = fft_corr_gpu(
                                bbdata_A['tiedbeam_baseband'][
                                    iifreq,
                                    iipol,
                                    start: stop,
                                ],
                                bbdata_A['tiedbeam_baseband'][
                                    iifreq,
                                    jjpol,
                                    start: stop,
                                ])
                            auto_vis[iifreq, pointing, iipol, jjpol, jjscan, :] = torch.concat(
                                (_vis[:max_lag+1], _vis[-max_lag:]))
    return auto_vis

def crosscorr_core_vectorized_gpu(
    bbdata_A: BBData,
    bbdata_B: BBData,
    t_a: np.ndarray,
    window: Union[np.ndarray, int],
    R: Union[np.ndarray, float],
    calc_results: IMReader,
    DM: float,
    index_A: int=0,
    index_B: int=1,
    sample_rate: float=2.56,
    max_lag: Optional[int]=None,
    n_pol: int=2,
    complex_conjugate_convention: int=-1,
    intra_channel_sign: int=1,
):
    """
    **only works for sources with a constant window as a function of frequency (can still vary with pointing and scan)**
    ** also currently only tested for steady sources with constant R.
    inputs:
    bbdata_A - telescope A data (sliced into a frequency chunk)
    bbdata_B - telescope B data (sliced into a frequency chunk)
    t_a[i,j] - starting frames at ith frequency, for jth time chunk, for telescope A
    window - np.array of size (nscans) containing integer numbers, each element is the length of scan window in frames
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
    #assert (type(R)==float or type(R)==int or R.shape=(n_freq,n_scan)), 'R needs to either be a number (1 for steady sources) or a numpy array of size (nfreq, nscan)'
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(t_a, axis=-1)
    # SA: basing this off of how the data is arranged now, may want to change
    n_pointings = bbdata_A["tiedbeam_baseband"].shape[1] // 2

   
    ctime_diff=bbdata_A["time0"]["ctime"]-bbdata_B["time0"]["ctime"]
    ctime_offset_diff=bbdata_A["time0"]["ctime_offset"]-bbdata_B["time0"]["ctime_offset"]
    delta_A_B=ctime_diff-ctime_offset_diff

    #assert f0==bbdata_B.index_map["freq"]["centre"], "frequency values need to be same"

    # initialize output autocorrelations and cross correlations
    if max_lag is None:
        # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.
        max_lag = 100
        
    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    cross = torch.zeros(vis_shape, dtype=torch.complex64)

    for pointing in range(n_pointings):
        # require user to have "well-ordered" bbdata in frequency (iifreqA=iifreqB)
        # frequency centers in MHz # array of length 1024
        #shape is (nfreq)
        start = time.time()
        baseband_a=torch.as_tensor(np.array(bbdata_A['tiedbeam_baseband']))
        baseband_a=baseband_a.to(device)
        baseband_b=torch.as_tensor(np.array(bbdata_B['tiedbeam_baseband']))
        baseband_b=baseband_b.to(device)
        f0=torch.as_tensor(np.array(bbdata_B.index_map["freq"]["centre"]))
        f0=f0.to(device)
        end = time.time()
        print(f"convert to torch: {end-start}")

        for jjscan in range(n_scan):
            if type(window)==int:
                window_jjscan=window
            else:
                window_jjscan=window[jjscan]

            t_a_indices = t_a[:, jjscan]  # array of length 1024
            t0_a = bbdata_A["time0"]["ctime"][:] + t_a_indices * (sample_rate*1e-6)  # array of length 1024

            start_times = Time(
                t0_a,
                val2=bbdata_A["time0"]["ctime_offset"][:],
                format="unix",
                precision=9,
            )
            start_times._set_scale('tai')
            dt_vals=(sample_rate * 1e-6 * (t_a_indices[:, np.newaxis] + 1 + np.arange(window_jjscan)))

            geodelays_flattened = calc_results.retarded_baseline_delay(
                ant1=index_A, ant2=index_B, time=start_times, src=pointing, delay_sign=0, self_consistent=False,
                frame_dt=dt_vals
            )
            geodelays = geodelays_flattened.reshape(dt_vals.shape)
            # Fringestopping B -> A

            scan_a_cd, scan_b_fs_cd = get_aligned_scans_gpu(
                baseband_a, baseband_b, f0,t_a_indices, window_jjscan, geodelays,delta_A_B,
                complex_conjugate_convention=complex_conjugate_convention, intra_channel_sign=intra_channel_sign, sample_rate=sample_rate
            )

            #freeing up vram: want to delete original rather than deleting a view of the tensor (i.e. a reference, which is what is seen in external function calls)  
            del baseband_a
            del baseband_b
            del f0

            print("scans are aligned")
            #######################################################
            ######### intrachannel de-dispersion ##################
            if DM!=0: #save computation time
                print("not yet implemented for pulses")

            #######################################################
            # Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on-signal ######
            if type(R)==int: #should be 1 for steady sources
                r_jjscan=R
            elif len(np.unique(R[:,jjscan]))==1:
                r_jjscan=R[0,jjscan]

            else: # "on" window varies as a function of frequency (e.g. pulsar)
                r_jjscan=R[:,jjscan] #np array of size (nfreq)
                if len(np.unique(r_jjscan))==1:
                    r_jjscan=r_jjscan[0]

            if type(r_jjscan)==int or type(r_jjscan)==float:
                r_ij=r_jjscan
                start = int((window_jjscan - window_jjscan*r_ij) // 2)
                stop = int((window_jjscan + window_jjscan*r_ij) // 2)
                #######################################################
                ########## cross-correlate the on-signal ##############
                for pol_0 in range(n_pol):
                    for pol_1 in range(n_pol):
                        if pol_0 == pol_1:
                            print("FFT CORR")
                            _vis = fft_corr_gpu(
                                scan_a_cd[:, pol_0, start:stop],
                                scan_b_fs_cd[:, pol_1, start:stop])
                            cross[:, pointing, pol_0, pol_1, jjscan, :] = torch.concat(
                                (_vis[:,:max_lag+1], _vis[:,-max_lag:]),dim=-1)
            else:
                for r_ij in r_jjscan:
                    start = int((window_jjscan - window_jjscan*r_ij) // 2)
                    stop = int((window_jjscan + window_jjscan*r_ij) // 2)
                    #######################################################
                    ########## cross-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            if pol_0 == pol_1:
                                _vis = fft_corr_gpu(
                                    scan_a_cd[:, pol_0, start:stop],
                                    scan_b_fs_cd[:, pol_1, start:stop])
                                cross[:, pointing, pol_0, pol_1, jjscan, :] = torch.concat(
                                    (_vis[:,:max_lag+1], _vis[:,-max_lag:]),dim=-1)

    return cross.cpu().numpy()

def get_aligned_scans_gpu(
    baseband_a: torch.Tensor, 
    baseband_b: torch.Tensor, 
    f0: torch.Tensor,
    t_a_index: np.ndarray, 
    wij:np.ndarray, 
    tau:np.ndarray,
    delta_A_B: float,
    complex_conjugate_convention: int=-1, 
    intra_channel_sign:int=1, 
    sample_rate:float=2.56):
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.

    baseband_a : torch.tensor
        BBData['tiedbeam_baseband'] tensor, with arbitrary frequency coverage.

    baseband_b : torch.tensor
        A BBData['tiedbeam_baseband], with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    t_a_index : np.array of shape (1024)
        An array of indices corresponding to the start frames for telescope A

    w_ij : window length. Should be an integer, and brownie points for a good FFT length.

    tau : np.array (nfreq, n_frame) of dtype np.float
        A delay in microseconds to apply to BBData_b, corresponding to the geometric delay.
        The first index is the delay evaluated at time t_ij_a

    freq_index : int

    Returns
    -------
    aligned_a : np.array
        A dual-pol scan of shape (2,w_ij)

    aligned_b : np.array
        A dual-pol scan of shape (2,w_ij)

    newstart: int
        Number of frames by which we need to shift t_a_ij in order to ensure t_a_ij+geodelay is contained within bbdata_B. Note that in the event that geodelay is positive, newstart will always be 0 (assuming the user has chosen t_a_ij such that the unix time is in both datasets)

    Super technical note on floor vs round: it doesn't matter AS LONG AS you do a sub-sample rotation (or better yet, a frac samp correction)! Suppose your total delay is 10.6 frames.
    - You can round to 11 frames. You should keep track that you rounded to 11, and then do frac samp -0.4.
    - You can floor to 10 frames, you should keep track that you floored to 10, and then do frac samp +0.6.

    Answer should be the same either way -- as long as you do the frac samp correction!
    After doing the integer part (shift by either 10 or 11 frames), we need to apply a phase rotation. Note that exp(2j*np.pi*channel_center * -0.4/(2.56us) = exp(2j*np.pi*channel_center * +0.6/(2.56us), for the exact frequency corresponding to channel center, but not for any of the other frequencies that do not satisfy f = 800 - (N * 0.390625 MHz) for integers N -- this is the narrowband approximation. We experience some de-correlation near the band edges, which is why we use fractional sample correction in this code.
    """

    time_we_want_at_b = tau[:, 0]  # us
    a_shape = list(baseband_a.shape)
    a_shape[-1] = wij

    start = time.time()
    aligned_a = torch.zeros(a_shape, dtype=baseband_a.dtype)
    # TODO vectorize
    if len(np.unique(t_a_index))==1:
        aligned_a[:, ...] = baseband_a[:, ...,
                                                          t_a_index[0]:t_a_index[0] + wij]
    else:
        for i in range(len(t_a_index)):
            aligned_a[i, ...] = bbdata_A['tiedbeam_baseband'][i, ...,
                                                            t_a_index[i]:t_a_index[i] + wij]
    end = time.time()
    print(f"Creating aligned_a: {end-start}")

    # aligned_a = bbdata_A['tiedbeam_baseband'][freq_id,...,t_a_index:t_a_index + wij]
    # initialize aligned B array
    start=time.time()
    aligned_b = torch.zeros(
        baseband_a.shape, dtype=baseband_a.dtype)
    aligned_b=aligned_b.to(device)
    end = time.time()
    print(f"moving torch b: {end-start}")

    # calculate the additional offset between A and B in the event that the (samples points of) A and B are misaligned in absolute time by < 1 frame
    # i.e. to correctly fringestop, we must also account for a case such as:
    ## A:    |----|----|----|----|----| ##
    ## B: |----|----|----|----|----|    ##

    # TODO vectorize
    int_delay = np.array([int(np.round((timeb*1e-6 - delta) / (sample_rate*1e-6)))
                         for timeb, delta, in zip(time_we_want_at_b, delta_A_B)])
    # frame number closest to start time
    start_index_we_want_at_b = t_a_index+int_delay

    # account for case where t_a_index+geodelay < 0 (i.e. signal arrives at telescope B before start of data acquision)
    start_index_we_have_at_b = np.array(
        [np.max([start, 0]) for start in start_index_we_want_at_b])
    # if index_we_have_at_b is negative, this will be the amount we need to cushion our output data by
    pad_index_b = start_index_we_have_at_b-start_index_we_want_at_b
    # TODO vectorize -- for pad, start in zip(pad_index_b, start_index_we_have_at_b)] is slow
    start = time.time()
    w_pad = wij - pad_index_b
    ntime_start = baseband_b.size()[-1] - start_index_we_have_at_b
    new_wij = np.minimum(w_pad, ntime_start)
    # new_wij = np.array([np.min([wij-pad, bbdata_B.ntime-start])
    #                    for pad, start in zip(pad_index_b, start_index_we_have_at_b)])
    end = time.time()
    print(f"Creating new_wij: {end-start}")
    # if you are missing half the data, multiply by 2.
    correction_factor = wij / new_wij
    if correction_factor.any() > 2:
        # warn the user that the boundary conditions are sketch if we are missing e.g. more than half the data.
        print("warning: based on specified start time and scan length, over half the data is missing from telescope XX.")

    start = time.time()
    for i in range(len(pad_index_b)):
        aligned_b[i, ..., pad_index_b[i]:pad_index_b[i]+new_wij[i]] = \
            baseband_b[i, ...,start_index_we_have_at_b[i]:start_index_we_have_at_b[i]+new_wij[i]] * correction_factor[i]
    end = time.time()
    print(f"updating aligned_b: {end-start}")

    # multiply by the correction factor to ensure that a steady source, when correlated, has the correct flux corresponding to the desired w_ij, even when we run out of data.
    aligned_b = aligned_b[..., :wij]

    time_we_have_at_b = (delta_A_B+int_delay*sample_rate*1e-6)  # s
    sub_frame_tau = np.array([tau[i, :wij] - time_b*1e6 for time_b, i in zip(
        time_we_have_at_b, range(len(tau)))])  # sub-frame delay at start time in mircoseconds
    start = time.time()
    sub_frame_tau=torch.as_tensor(sub_frame_tau)
    sub_frame_tau=sub_frame_tau.to(device)
    end = time.time()
    print(f"creating sub_frame_tau: {end-start}")

    start = time.time()
    aligned_b = frac_samp_shift_gpu(aligned_b,
                                           f0=f0,
                                           sub_frame_tau=sub_frame_tau,
                                           complex_conjugate_convention=complex_conjugate_convention,
                                           intra_channel_sign=intra_channel_sign,
                                           sample_rate=sample_rate)
    end = time.time()
    print(f"Running frac_samp_shift_vectorized: {end-start}")
    aligned_a=aligned_a.to(device)
    print("moving aligned")
    return aligned_a, aligned_b


def frac_samp_shift_gpu(data, f0, sub_frame_tau, complex_conjugate_convention, intra_channel_sign, sample_rate=2.56):
    n = data.shape[-1]
    f = torch.fft.fftfreq(n, sample_rate)
    f=f.to(device)
    transfer_func = torch.exp(intra_channel_sign*2j * np.pi * f[np.newaxis,:] * torch.median(sub_frame_tau,dim=-1).values[:,np.newaxis])  # apply dphi/dfreq
    del f #free up vram
    output= torch.fft.ifft(
            torch.fft.fft(data) * transfer_func[:,np.newaxis]) * (torch.exp(complex_conjugate_convention*2j * np.pi * f0[:,np.newaxis] * sub_frame_tau))[:, np.newaxis, :]   # apply phi
    del transfer_func
    return output