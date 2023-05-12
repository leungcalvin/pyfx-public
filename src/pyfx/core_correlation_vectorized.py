import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
import time

from pyfx.core_math import fft_corr
from pyfx.core_math import max_lag_slice
import collections

#enable type hints for static tools
from baseband_analysis.core.bbdata import BBData
from difxcalc_wrapper.io import IMReader
from typing import Optional, Tuple, Union

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)

def autocorr_core_vectorized(
    DM: float, 
    bbdata_a: BBData,
    t_a: np.ndarray,
    window: Union[np.ndarray, int],
    R: Union[np.ndarray, float],
    max_lag: int,
    n_pol: int=2):
    """Auto-correlates data and downselects over lag
    Inputs:
    -------
    DM - the DM with which we de-smear the data before the final gating. for continuum sources, set dispersion measure to 0.
    bbdata_a - baseband data. Needs to have "tiedbeam_baseband" data of size (nfreq,npointing*npol.ntime)
    t_a[i,j] - start times at ith frequency, for jth time chunk, for telescope A. Upper layer should ensure that this is of size (nfreq,nscan). 
    window[j] - integer or np.array of size (nscan) holding length of time chunk window in frames. Upper layer should assert that this is of size (nscan).
    R[i,j] - float or np.array of size (nfreq,nscan). zFraction of time chunk (defines pulse window). Upper layer should ensure that this is less than 1, and that this is of size nxm.
    max_lag - maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data). Outer layer of the code should check that this is less than 1/2 of the window size times R[i,j]. 
    n_pol - number of polarizations in data
    Outputs:
    -------
    auto_vis - array of autocorrelations with shape (nfreq, npointing, npol, npol, nscan,nlag)

    """
    n_freq = len(bbdata_a.freq)
    n_scan = np.size(t_a, axis=-1)
    n_pointings = bbdata_a["tiedbeam_baseband"].shape[1] // n_pol

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    auto_vis = np.zeros(vis_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
    f0 = bbdata_a.index_map["freq"]["centre"] #shape is (nfreq)

    
    for jjscan in range(n_scan):
        if type(window)==int:
            window_jjscan=window
        else:
            window_jjscan=window[jjscan]

        t_a_indices = t_a[:, jjscan]  # array of length nfreq

        if isinstance(R,collections.abc.Sized)==False:#should be 1 for continuum sources                    
            r_jjscan=R
        elif len(np.unique(R[:,jjscan]))==1:
            r_jjscan=R[0,jjscan]

        else: # "on" window varies as a function of frequency (e.g. pulsar)
            r_jjscan=R[:,jjscan] #np array of size (nfreq)

        if isinstance(r_jjscan,collections.abc.Sized)==False and len(np.unique(t_a_indices))==1:
            # the simplest case, e.g. continuum source, we save time by completely vectorizing over frequency
            for pointing in range(n_pointings):
                for iipol in range(n_pol):
                    for jjpol in range(n_pol):
                        r_ij=r_jjscan
                        start = int((window_jjscan - window_jjscan*r_ij) // 2)+t_a_indices[0]
                        stop = int((window_jjscan + window_jjscan*r_ij) // 2)+t_a_indices[0]
                        scan_a_cd = intrachannel_dedisp_vectorized(bbdata_a['tiedbeam_baseband'][:,:,start:stop], DM, f0=f0)
                        _vis = fft_corr(scan_a_cd[:,iipol,:],scan_a_cd[:,jjpol,:])
                        auto_vis[:, pointing, iipol, jjpol, jjscan, :] = np.concatenate((_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
                    
        else:
            # if the start and stop times are not the same as a function of frequency, there is no easy way of vectorizing this. 
            # We can be smarter about this based on the window size, but for now
            # be lazy and resort to a for loop 
        
            ## note: this is very slow
            scan_a_cd = intrachannel_dedisp_vectorized(bbdata_a['tiedbeam_baseband'][:,:,:,], DM, f0=f0)
            for pointing in range(n_pointings):
                for iipol in range(n_pol):
                    for jjpol in range(n_pol):
                        if isinstance(r_jjscan,collections.abc.Sized)==False:
                            r_jjscan=np.full(n_freq,r_jjscan)
                        for iifreq,r_ij in enumerate(r_jjscan):
                            start = int((window_jjscan - window_jjscan*r_ij) // 2)+t_a_indices[iifreq]
                            stop = int((window_jjscan + window_jjscan*r_ij) // 2)+t_a_indices[iifreq]
                            _vis = fft_corr(scan_a_cd[iifreq,iipol,:],scan_a_cd[iifreq,jjpol,:])
                            auto_vis[iifreq, pointing, iipol, jjpol, jjscan, :] = np.concatenate(
                                (_vis[:max_lag+1], _vis[-max_lag:]))
                        
    return auto_vis

def crosscorr_core_vectorized(
    bbdata_a: BBData,
    bbdata_b: BBData,
    t_a: np.ndarray,
    window: Union[np.ndarray, int],
    R: Union[np.ndarray, float],
    calc_results: IMReader,
    DM: float,
    index_A: int,
    index_B: int,
    max_lag: int,
    sample_rate: float=2.56,
    n_pol: int=2,
    complex_conjugate_convention: int=-1,
    intra_channel_sign: int=1,
    fast: bool=True
):
    """Fringestops, coherently dedisperses, and cross correlates data 
    Inputs:
    -------
    bbdata_a - telescope A baseband data 
    bbdata_b - telescope B baseband data. Data must be "weel-ordered" in frequency (iifreqA=iifreqB). Frequency centers must also be in Mhz. 
    t_a[i,j] - array of integers corresponding to start frames at ith frequency, for jth time chunk, for telescope A. Upper layer should ensure that this is of size nxm. 
    window - np.array of size (nscans) containing integer numbers, each element is the length of scan window in frames
    R[i,j] - fraction of time chunk (defines pulse window). For continuum sources, R=1 ("on" window = full window).Upper layer should ensure that this is less than 1.
    calc_results - difxcalc-wrapper IMReader object, which is used to calculate geometric delays
    DM - the DM with which we de-smear the data before the final gating. for continuum sources, set dispersion measure to 0.
    index_A - where telescope A corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    index_B - where telescope B corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    max_lag - maximum (absolute value) lag (in frames) for correlations (useful for very long time series data)
    sample_rate - rate at which data is sampled in microseconds
    n_pol - number of polarizations 
    complex conjugate convention - should be a value of -1 if the baseband data is complex conjugated with respect to the sky, 1 otherwise
    intra_channel_sign - a sign to account for a reflection of frequencies about zero (e.g. in iq/baseband data). Should be -1 if frequencies within a channel are reflected about 0, 1 otherwise. 
    
    Outputs:
    -------
    cross - array of cross correlations with shape (nfreq, npointing, npol, npol, nscan,nlag)
    """
    n_freq = len(bbdata_a.freq)
    n_scan = np.size(t_a, axis=-1)
    # SA: basing this off of how the data is arranged now, may want to change
    n_pointings = bbdata_a["tiedbeam_baseband"].shape[1] // n_pol

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    cross = np.zeros(vis_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
    f0 = bbdata_b.index_map["freq"]["centre"] #shape is (nfreq)
    for npointing in range(n_pointings):
        for jjscan in range(n_scan):
            if type(window)==int:
                window_jjscan=window
            else:
                window_jjscan=window[jjscan]

            t_a_indices = t_a[:, jjscan]  # array of length 1024
            t0_a = bbdata_a["time0"]["ctime"][:]
            t0_a_offset=bbdata_a["time0"]["ctime_offset"][:]+ t_a_indices * (sample_rate*1e-6)  # array of length 1024

            start_times = Time(
                t0_a,
                val2=t0_a_offset,
                format="unix",
                precision=9,
            )
            # using telescope A times as reference time
            if fast==True:
                dt_vals=(sample_rate * 1e-6 * (t_a_indices[:, np.newaxis] + 1 + np.arange(window_jjscan)))
                geodelays_flattened = calc_results.retarded_baseline_delay(
                    ant1=index_A, ant2=index_B, time=start_times, src=npointing, delay_sign=0, self_consistent=False,
                    frame_dt=dt_vals
                )
                geodelays = geodelays_flattened.reshape(dt_vals.shape)
            else:
                geodelays=np.zeros((1024,window_jjscan),dtype=float)
                for i in range(n_freq):
                    query_times = start_times[i] + sample_rate*1e-6 * un.s * (t_a_indices[i]+np.arange(window_jjscan))
                    geodelays[i,:]=calc_results.retarded_baseline_delay(
                        ant1=index_A, ant2=index_B, time=query_times, src=npointing,delay_sign=0,self_consistent=False
                    )

            # Fringestopping B -> A
            scan_a, scan_b_fs = get_aligned_scans_vectorized(
                bbdata_a, bbdata_b, t_a_indices, window_jjscan, geodelays,
                complex_conjugate_convention=complex_conjugate_convention, intra_channel_sign=intra_channel_sign, sample_rate=sample_rate,
                npointing=npointing,n_pol=n_pol
            )

            #######################################################
            ######### intrachannel de-dispersion ##################
            scan_a_cd = intrachannel_dedisp_vectorized(scan_a, DM, f0=f0)
            scan_b_fs_cd = intrachannel_dedisp_vectorized(scan_b_fs, DM, f0=f0)

            #######################################################
            # Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on-signal ######
            if type(R)==int: #should be 1 for continuum sources
                r_jjscan=R

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
                        _vis = fft_corr(
                            scan_a_cd[:, pol_0, start:stop],
                            scan_b_fs_cd[:, pol_1, start:stop])
                        cross[:, npointing, pol_0, pol_1, jjscan, :] = np.concatenate(
                            (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
            else:
                for r_ij in r_jjscan:
                    start = int((window_jjscan - window_jjscan*r_ij) // 2)
                    stop = int((window_jjscan + window_jjscan*r_ij) // 2)
                    #######################################################
                    ########## cross-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            if pol_0 == pol_1:
                                _vis = fft_corr(
                                    scan_a_cd[:, pol_0, start:stop],
                                    scan_b_fs_cd[:, pol_1, start:stop])
                                cross[:, npointing, pol_0, pol_1, jjscan, :] = np.concatenate(
                                    (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)

    return cross

def get_aligned_scans_vectorized(
    bbdata_a: BBData, 
    bbdata_b: BBData, 
    t_a_index: np.ndarray, 
    wij: int, 
    tau: np.ndarray, 
    complex_conjugate_convention: int=-1, 
    intra_channel_sign: int=1, 
    sample_rate: float =2.56,
    npointing:int=0,
    n_pol:int=2):
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.
    Inputs:
    -------
    bbdata_a : BBData
        A BBData object, with arbitrary frequency coverage.

    bbdata_b : BBData
        A BBData object, with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    t_a_index : np.array of shape (1024)
        An array of indices corresponding to the start frames for telescope A

    w_ij : int
        window length. Should be an integer, and brownie points for a good FFT length.

    tau : np.array (nfreq, n_frame) of dtype np.float
        A delay in microseconds to apply to BBData_b, corresponding to the geometric delay.
        The first index is the delay evaluated at time t_ij_a

    freq_index : int

    Outputs:
    -------
    aligned_a : np.array
        A dual-pol scan of shape (2,w_ij)

    aligned_b : np.array
        A dual-pol scan of shape (2,w_ij)

    Super technical note on floor vs round: it doesn't matter AS LONG AS you do a sub-sample rotation (or better yet, a frac samp correction)! Suppose your total delay is 10.6 frames.
    - You can round to 11 frames. You should keep track that you rounded to 11, and then do frac samp -0.4.
    - You can floor to 10 frames, you should keep track that you floored to 10, and then do frac samp +0.6.

    Answer should be the same either way -- as long as you do the frac samp correction!
    After doing the integer part (shift by either 10 or 11 frames), we need to apply a phase rotation. Note that exp(2j*np.pi*channel_center * -0.4/(2.56us) = exp(2j*np.pi*channel_center * +0.6/(2.56us), for the exact frequency corresponding to channel center, but not for any of the other frequencies that do not satisfy f = 800 - (N * 0.390625 MHz) for integers N -- this is the narrowband approximation. We experience some de-correlation near the band edges, which is why we use fractional sample correction in this code.
    """

    time_we_want_at_b = tau[:, 0]  # us
    a_shape = list(bbdata_a['tiedbeam_baseband'][:,npointing:npointing+n_pol,:].shape)
    a_shape[-1] = wij

    aligned_a = np.zeros(a_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
    # TODO vectorize
    if len(np.unique(t_a_index))==1:
        aligned_a[:, ...] = bbdata_a['tiedbeam_baseband'][:,npointing:npointing+n_pol,
                                                          t_a_index[0]:t_a_index[0] + wij]
    else:
        for i in range(len(t_a_index)):
            aligned_a[i, ...] = bbdata_a['tiedbeam_baseband'][i,npointing:npointing+n_pol,
                                                            t_a_index[i]:t_a_index[i] + wij]

    # aligned_a = bbdata_a['tiedbeam_baseband'][freq_id,...,t_a_index:t_a_index + wij]
    # initialize aligned B array
    aligned_b = np.zeros(
        bbdata_b['tiedbeam_baseband'][:,npointing:npointing+n_pol,:].shape, dtype=bbdata_b['tiedbeam_baseband'].dtype)
    # calculate the additional offset between A and B in the event that the (samples points of) A and B are misaligned in absolute time by < 1 frame
    # i.e. to correctly fringestop, we must also account for a case such as:
    ## A:    |----|----|----|----|----| ##
    ## B: |----|----|----|----|----|    ##
    
    t_a= Time(
        bbdata_a["time0"]["ctime"][:],
        val2=bbdata_a["time0"]["ctime_offset"][:],
        format="unix",
        precision=9)
    t_b= Time(
        bbdata_b["time0"]["ctime"][:],
        val2=bbdata_b["time0"]["ctime_offset"][:],
        format="unix",
        precision=9)
    delta_A_B=(t_b-t_a).to_value('sec') #this is actually not that time consuming for 1024 frequencies

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
    w_pad = wij - pad_index_b
    ntime_start = bbdata_b.ntime - start_index_we_have_at_b
    new_wij = np.minimum(w_pad, ntime_start)
    new_wij = np.array([np.min([wij-pad, bbdata_b.ntime-start])
                        for pad, start in zip(pad_index_b, start_index_we_have_at_b)])
    # if you are missing half the data, multiply by 2.
    
    correction_factor =wij / new_wij
    if correction_factor.any() > 2:
        # warn the user that the boundary conditions are sketch if we are missing e.g. more than half the data.
        print("warning: based on specified start time and scan length, over half the data is missing from telescope XX.")

    for i in range(len(pad_index_b)):
        aligned_b[i, ..., pad_index_b[i]:pad_index_b[i]+new_wij[i]] = \
            bbdata_b['tiedbeam_baseband'][i, ...,start_index_we_have_at_b[i]:start_index_we_have_at_b[i]+new_wij[i]] * correction_factor[i]
    # multiply by the correction factor to ensure that a continuum source, when correlated, has the correct flux corresponding to the desired w_ij, even when we run out of data.
    aligned_b = aligned_b[..., :wij]

    time_we_have_at_b = (delta_A_B+int_delay*sample_rate*1e-6)  # s
    sub_frame_tau = np.array([tau[i, :wij] - time_b*1e6 for time_b, i in zip(
        time_we_have_at_b, range(len(tau)))])  # sub-frame delay at start time in mircoseconds

    aligned_b = frac_samp_shift_vectorized(aligned_b,
                                           f0=bbdata_b.index_map["freq"]["centre"][:],
                                           sub_frame_tau=sub_frame_tau,
                                           complex_conjugate_convention=complex_conjugate_convention,
                                           intra_channel_sign=intra_channel_sign,
                                           sample_rate=sample_rate)
    return aligned_a, aligned_b

#### faster option is with gpus
def frac_samp_shift_vectorized(
    data:np.ndarray, 
    f0:np.ndarray, 
    sub_frame_tau:np.ndarray, 
    complex_conjugate_convention:int=-1, 
    intra_channel_sign:int=1, 
    sample_rate: float=2.56):
    """
    Coherently shifts data within a channel via a fractional phase shift of the form exp(2j*pi*f*sub_frame_tau).
    Inputs:
    -------    
    data : np.ndarray of shape (nfreq,npol*npointing,ntime)

    f0 : frequency channel center.

    sample_rate : sampling rate of data in microseconds

    sub_frame_tau: np.array of shape (ntime), sub-frame delay in us

    complex_conjugate_convention: a sign to account for the fact that the data may be complex conjugated

    intra_channel_sign: a sign to account for a reflection of frequencies about zero (e.g. in iq/baseband data)


    """
    # glorified element wise multiplication
    #data will be of shape (nfreq,npol,ntime)
    n = data.shape[-1]
    f = np.fft.fftfreq(n, sample_rate)
    # transfer_func is now of shape (nfreq,ntime)
    transfer_func = np.exp(intra_channel_sign*2j * np.pi * f[np.newaxis,:] * np.median(sub_frame_tau,axis=-1)[:,np.newaxis])  # apply dphi/dfreq
    return np.fft.ifft(
            np.fft.fft(data, axis=-1) * transfer_func[:,np.newaxis,], axis=-1
        ) * (np.exp(complex_conjugate_convention*2j * np.pi * f0[:,np.newaxis] * sub_frame_tau))[:, np.newaxis, :]   # apply phi

def intrachannel_dedisp_vectorized(
    data: np.ndarray,
    DM: float,
    f0: np.ndarray,
    sample_rate: float = 2.56):
    """Intrachannel dedispersion: brings data to center of channel.
    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook, but ONLY the last term (proportional to f^2), not the other two terms (independent of f and linearly proportional to f respectively).
    Inputs:
    -------
    data : np.ndarray of shape (nfreq,npol*npointing,ntime)
    f0 : np.ndarray of shape (nfreq) holding channel centers.
    sample_rate : sampling rate of data in microseconds
    Outputs:
    -------
    data: np.ndarray of shape (nfreq,npol*npointing,ntime)
    """
    if DM==0: #save computation time
        return data
    else:        
        n = data.shape[-1]
        f = np.fft.fftfreq(n) * sample_rate
        transfer_func = np.exp(-2j * np.pi * K_DM * DM * f[np.newaxis,:]**2 / f0[:,np.newaxis]**2 / (f[np.newaxis,:] + f0[:,np.newaxis]))  
        data = np.fft.ifft(np.fft.fft(data, axis=-1) * transfer_func[:,np.newaxis,:],axis=-1)
        return data
