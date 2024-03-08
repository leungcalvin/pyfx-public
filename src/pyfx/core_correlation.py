"""
Fringestops station B to station A and cross correlates baseband data from station A and B. 
The "core" module that should be called by the "outer layer" corr_job.py
All bugs are the responsibility of Shion Andrew   
"""

import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
import time
from pyfx.fft_corr import basic_correlator #subframe_signal_to_noise_search_correlator as basic_correlator # could swap out correlator here
from pyfx import config
import collections
from pycalc11 import Calc
from baseband_analysis.core.bbdata import BBData
from typing import Optional, Tuple, Union
import logging

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
MAX_FRAC_SAMP_LENGTH=32187 #maximum FFT length, chosen to keep delay rate drift (on Earth) within 1/10th of a frame 

def autocorr_core(
    DM: float, 
    bbdata_a: BBData,
    t_a: np.ndarray,
    window: np.ndarray,
    R: np.ndarray,
    max_lag: Optional[int]=None,
    n_pol: int=2,
    zp: bool=True,
    ) -> np.ndarray:
    """Auto-correlates data and downselects over lag.

    Parameters
    ----------
    DM : float
        The DM with which the zeroth pointing of the data is de-smeared before the final gating. for continuum sources, set dispersion measure to 0.

    bbdata_a : BBData object
        At bare minimum, needs to have "tiedbeam_baseband" data of size (nfreq, npointing*npol, ntime).
    
    t_a : np.ndarray of int of shape (nfreq, npointing, nscan).    
        start index of the integration, relative to bbdata_a['time0']['ctime'] in units of 2.56 microsec, as a function of frequency channels, pointing index, and time in units of :window: (i.e. scan number).
    
    window : np.ndarray of int of shape (npointing, nscan).
        duration of the scan, in units of 2.56 microsec, as a function of pointing and time (i.e. scan number).
    
    R : np.ndarray of float of shape (nfreq, npointing, nscan).
        Fraction R <= 1 of the scan, that gets down-selected before integration. In other words, we integrate between t_a + window // 2 +- r * window / 2

    max_lag : int (Optional)
        maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data). TODO: Outer layer of the code should check that this is less than 1/2 of the window size times R.
        set this to 20 for a good balance between space efficiency and good noise statistics.
    
    n_pol : int
        number of polarizations in data -- always 2.

    Returns
    -------
    auto_vis - array of autocorrelations with shape (nfreq, npointing, npol, npol, 2 * nlag + 1, nscan)

    """
    if max_lag is None: #set to default
        max_lag = config.CHANNELIZATION['nlags']
    n_freq = bbdata_a.nfreq
    n_scan = np.size(t_a, axis=-1)
    n_pointings = bbdata_a["tiedbeam_baseband"].shape[1] // n_pol
    
    # convert all nans to 0s
    bbdata_a['tiedbeam_baseband'][:]=np.nan_to_num(bbdata_a['tiedbeam_baseband'][:], nan=0, posinf=0, neginf=0)

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, 2 * max_lag + 1,n_scan)
    auto_vis = np.zeros(vis_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
    f0 = bbdata_a.index_map["freq"]["centre"] #shape is (nfreq)
    
    for kkpointing in range(n_pointings):
        for jjscan in range(n_scan):
            wij=int(window[kkpointing,jjscan])
            t_a_indices = t_a[:, kkpointing,jjscan]  # array of length 1024
            ## clip telescope A data ##
            a_shape = list(bbdata_a['tiedbeam_baseband'][:,kkpointing:kkpointing+n_pol,:].shape)
            a_shape[-1] = wij
            clipped_a = np.zeros(a_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
            if len(np.unique(t_a_indices))==1 and not zp:
                # go fast
                clipped_a[:, ...] = bbdata_a['tiedbeam_baseband'][:,kkpointing:kkpointing+n_pol,
                                                                t_a_indices[0]:t_a_indices[0] + wij]
            elif len(np.unique(t_a_indices)) > 1 and not zp:
                # go slower
                for i in range(len(t_a_indices)):
                    clipped_a[i, ...] = bbdata_a['tiedbeam_baseband'][i,kkpointing:kkpointing+n_pol,
                                                                    t_a_indices[i]:t_a_indices[i] + wij]
            elif zp:
                for i in range(len(t_a_indices)):
                    for j in range(n_pol):
                        clipped_a[i, j, :] = getitem_zp1d(bbdata_a['tiedbeam_baseband'][i,2*kkpointing + j],
                                                                    t_a_indices[i],
                                                                    t_a_indices[i] + wij)
                
            ######### intrachannel de-dispersion ##################
            scan_a_cd = intrachannel_dedisp(clipped_a, DM, f0=f0)
            r_jjscan=R[:,kkpointing,jjscan] #np array of size (nfreq)
            if len(np.unique(r_jjscan))==1:
                r_ij=r_jjscan[0]
                start = int((wij - wij*r_ij) // 2)
                stop = int((wij + wij*r_ij) // 2)
                #######################################################
                ########## auto-correlate the on-signal ##############
                for pol_0 in range(n_pol):
                    for pol_1 in range(n_pol):
                        _vis = basic_correlator(
                            scan_a_cd[:, pol_0, start:stop],
                            scan_a_cd[:, pol_1, start:stop], full_output = False)
                        auto_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                            (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
            else:
                for r_ij in r_jjscan:
                    start = int((wij - wij*r_ij) // 2)
                    stop = int((wij + wij*r_ij) // 2)
                    #######################################################
                    ########## auto-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            if pol_0 == pol_1:
                                _vis = basic_correlator(
                                    scan_a_cd[:, pol_0, start:stop],
                                    scan_a_cd[:, pol_1, start:stop],full_output = False)
                                auto_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                                    (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
         
    return auto_vis

def crosscorr_core(
    bbdata_a: BBData,
    bbdata_b: BBData,
    t_a: np.ndarray,
    window: np.ndarray,
    R: np.ndarray,
    pycalc_results: Calc,
    DM: float,
    index_A: int,
    index_B: int,
    max_lag: Optional[int]=None,
    sample_rate: float=2.56,
    n_pol: int=2,
    complex_conjugate_convention: int=-1,
    intra_channel_sign: int=1,
    weight: Optional[np.ndarray]=None,
    fast:bool=False,
    zp: bool=True,
    ) -> np.ndarray:
    """Fringestops, coherently dedisperses, and cross correlates data 
    Parameters
    ----------
    bbdata_a : BBData object
        At bare minimum, needs to have "tiedbeam_baseband" data of size (nfreq, npointing*npol, ntime).
    bbdata_b : 
        telescope B baseband data. Data must have matching index_map['freq'] as bbdata_a. index_map['freq']['centre'] must also be in MHz.
    t_a : np.ndarray of int of shape (nfreq, npointing, nscan).    
        start index of the integration, relative to bbdata_a['time0']['ctime'] in units of 2.56 microsec, as a function of frequency channels, pointing index, and time in units of :window: (i.e. scan number).
    
    window : np.ndarray of int of shape (npointing, nscan).
        duration of the scan, in units of 2.56 microsec, as a function of pointing and time (i.e. scan number).
    
    R : np.ndarray of float of shape (nfreq, npointing, nscan).
        Fraction R <= 1 of the scan, that gets down-selected before integration. In other words, we integrate between t_a + window // 2 +- r * window / 2
    
    DM : float
        The DM with which the zeroth pointing of the data is de-smeared before the final gating. for continuum sources, set dispersion measure to 0.
    
    n_pol : int
        number of polarizations in data -- always 2.
    
    pycalc_results :
        pycalc11 Calc object, which is used to calculate geometric delays. Calc object should be initialized outside of this function and driver should have already been run (i.e. ci.run_driver())
    
    index_A :
        where telescope A corresponds to in pycalc_results.
    
    index_B :
        where telescope B corresponds to in pycalc_results.

    max_lag : int (Optional)
        maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data). TODO: Outer layer of the code should check that this is less than 1/2 of the window size times R.
        set this to 20 for a good balance between space efficiency and good noise statistics.    

    sample_rate :
        rate at which data is sampled in microseconds
    
    n_pol :
        number of polarizations 
    
    complex conjugate convention :
        should be a value of -1 if the baseband data is complex conjugated with respect to the sky, 1 otherwise
    
    intra_channel_sign :
        a sign to account for a reflection of frequencies about zero (e.g. in iq/baseband data). Should be -1 if frequencies within a channel are reflected about 0, 1 otherwise. 
    
    fast : 
        if False, use astropy addition (high precision but slow) and subtraction to evaluate the geodelays as a function of time. 
        If True, use float addition/subtraction; this should always be fine (picosecond precision) so long as pycalc_results.times[0] is within ~100s of all timestamps at which delays are to be evaluated.

    weight : 
        array of shape (nfreq, nscan, npointings,ntime) that specifies what weighting to apply to the data **relative to the start time given by t_a**.
        The shape of weight[:,jjscan,kkpointing] should be window[jjscan,kkpointing]
        
    Outputs:
    -------
    cross_vis :
        array of cross_vis correlation visibilities with shape (nfreq, npointing, npol, npol, nlag,nscan)
    """
    if max_lag is None:
        max_lag = config.CHANNELIZATION['nlags']
    n_freq = len(bbdata_a.freq)
    n_scan = np.size(t_a, axis=-1)
    # SA: basing this off of how the data is arranged now, may want to change
    n_pointings = bbdata_a["tiedbeam_baseband"].shape[1] // n_pol
    n_freq_B=len(bbdata_b.freq)
    assert n_freq_B==n_freq, f"There appear to be {n_freq} frequency channels in telescope A and {n_freq_B} frequency channels in telescope B. Please pass in these bbdata objects with frequency channels aligned (i.e. nth index along the frequency axis should correspond to the *same* channel in telescope A and B)"
    vis_shape = (n_freq, n_pointings, n_pol, n_pol, 2 * max_lag + 1,n_scan)
    cross_vis = np.zeros(vis_shape, dtype=bbdata_a['tiedbeam_baseband'].dtype)
    f0 = bbdata_a.index_map["freq"]["centre"] #shape is (nfreq)
    f0_b = bbdata_b.index_map["freq"]["centre"] #shape is (nfreq)

    # convert all nans to 0s
    bbdata_a['tiedbeam_baseband'][:]=np.nan_to_num(bbdata_a['tiedbeam_baseband'][:], nan=0, posinf=0, neginf=0)
    bbdata_b['tiedbeam_baseband'][:]=np.nan_to_num(bbdata_b['tiedbeam_baseband'][:], nan=0, posinf=0, neginf=0)

    
    for kkpointing in range(n_pointings):
        for jjscan in range(n_scan):
            wij=window[kkpointing,jjscan]
            t_a_indices = t_a[:, kkpointing, jjscan]  # array of length 1024
            t0_a = bbdata_a["time0"]["ctime"][:]
    
            # using telescope A times as reference time
            t0_a_offset=bbdata_a["time0"]["ctime_offset"][:] + t_a_indices * (sample_rate*1e-6)  # array of length 1024
            
            #start time of reference frequency channel
            ref_start_time = Time(
                t0_a[0],
                val2=t0_a_offset[0],
                format="unix",
                precision=9,
            )
            delta_ctime=t0_a-t0_a[0]
            delta_ctime_offset=t0_a_offset-t0_a_offset[0] #difference between reference start time and nth freqeucny start time
            
            if fast: #for the impatient
                delta_t=delta_ctime+delta_ctime_offset
                dt_vals=sample_rate * 1e-6 * np.arange(wij)+delta_t[:,np.newaxis] #nfreq,nframe
                dt_vals0=(ref_start_time-pycalc_results.times[0]).sec #should always be <1s. 
                delays_flattened=pycalc_results.delays_dt(dt_vals0+dt_vals.flatten())
                geodelays_flattened=delays_flattened[:,0,index_B,:]-delays_flattened[:,0,index_A,:] #units of seconds
                geodelays = geodelays_flattened.reshape(dt_vals.shape)*1e6 #microseconds #nfreq,nframe
            else: #for the paranoid
                t0_a_offset=bbdata_a["time0"]["ctime_offset"][:] # array of length 1024
                start_times = Time(
                    t0_a,
                    val2=t0_a_offset,
                    format="unix",
                    precision=9,
                ) #these are the pure ctime start times from the data 
                geodelays=np.zeros((1024,wij),dtype=float)
                for i in range(n_freq):
                    # the times we want to query for each frequency is an array of length wij times ranging from (ctime start times + t_a, ctime start times + t_a +w_ij)
                    query_times = start_times[i] + sample_rate*1e-6 * un.s * (t_a_indices[i]+np.arange(wij))
                    delays=pycalc_results.interpolate_delays(query_times)
                    geodelays[i,:]=(delays[:,0,index_B,0]-delays[:,0,index_A,0])*1e6

            # Fringestopping B -> A
            scan_a, scan_b_fs = get_aligned_scans(
                bbdata_a, bbdata_b, t_a_indices, wij, geodelays,
                complex_conjugate_convention=complex_conjugate_convention, intra_channel_sign=intra_channel_sign, sample_rate=sample_rate,
                npointing=kkpointing,n_pol=n_pol,zp=zp
            )

            #######################################################
            ######### intrachannel de-dispersion ##################
            scan_a_cd = intrachannel_dedisp(scan_a, DM, f0=f0)
            scan_b_fs_cd = intrachannel_dedisp(scan_b_fs, DM, f0=f0)

            ## weight the data
            if type(weight)==np.ndarray:
                scan_a_cd*=weight[:,np.newaxis,jjscan,kkpointing,:]
                scan_b_fs_cd*=weight[:,np.newaxis,jjscan,kkpointing,:]

            #######################################################
            # Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on-signal ######
            r_jjscan=R[:,kkpointing,jjscan] #np array of size (nfreq)

            if len(np.unique(r_jjscan))==1:
                #we can easily vectorize over frequency
                r_ij=r_jjscan[0]
                start = int((wij - wij*r_ij) // 2)
                stop = int((wij + wij*r_ij) // 2)
                if start==stop:
                    logging.warning("r_ij includes less than 1 frame: visibilities will be set to 0")
                else:
                    #######################################################
                    ########## cross-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            assert not np.isnan(np.min(scan_a_cd[:, pol_0, start:stop].flatten())), "Scan parameters have been poorly defined for telescope A. Please ensure there are no nans in the baseband data"
                            assert not np.isnan(np.min(scan_b_fs_cd[:, pol_0, start:stop].flatten())), "Scan parameters have been poorly defined for telescope B. Please ensure there are no nans in the baseband data"
                            _vis = basic_correlator(
                                scan_a_cd[:, pol_0, start:stop],
                                scan_b_fs_cd[:, pol_1, start:stop],
                                full_output = False)
                            cross_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                                (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
            else:
                #loop over frequency channel
                for freq in range(len(r_jjscan)):
                    r_ij=r_jjscan[freq]
                    start = int((wij - wij*r_ij) // 2)
                    stop = int((wij + wij*r_ij) // 2)
                    if start==stop:
                        logging.warning("r_ij includes less than 1 frame: visibilities for this channel will be set to 0")
                    else:
                        #######################################################
                        ########## cross-correlate the on-signal ##############
                        for pol_0 in range(n_pol):
                            for pol_1 in range(n_pol):
                                assert not np.isnan(np.min(scan_a_cd[freq, pol_0, start:stop].flatten())), "Scan parameters have been poorly defined for telescope A. Please ensure there are no nans in the baseband data"
                                assert not np.isnan(np.min(scan_b_fs_cd[freq, pol_0, start:stop].flatten())), "Scan parameters have been poorly defined for telescope B. Please ensure there are no nans in the baseband data"
                                _vis = basic_correlator(
                                    scan_a_cd[freq, pol_0, start:stop],
                                    scan_b_fs_cd[freq, pol_1, start:stop],
                                    full_output = False)
                                cross_vis[freq, kkpointing, pol_0, pol_1, :,jjscan] = np.concatenate(
                                    (_vis[:max_lag+1], _vis[-max_lag:]),axis=-1)

    return cross_vis

def getitem_zp1d(arr,start_want,stop_want):
    """Acts like arr[start_want:stop_want] but assumes start is strictly less than stop.

    It returns output with the properties that 
        1) width = stop_want - start_want 
        2) as if bbdata were zero-padded on the left and right to negative and positive infinity.

    Of course, no zero-padding actually takes place, to save memory. We implement this with casework:

    All out: stop_want < start_have OR stop_have < start_want
        We return all zeros
    Half in, data late: start_want < start_have < stop_want < stop_have 
        We zero-pad at the start of the output.
    Half in, data early: start_have < start_want < stop_have < stop_want 
        We zero-pad at the end of the output
    All in : start_have < start_want < stop_want < stop_have -- easy peasy.
    
    TODO: make this work over a given axis of an arbitrary np.ndarray
    """
    width = stop_want - start_want
    assert width >= 0, "Negative scan length not allowed; check your w_ij"
    out = np.zeros(dtype = arr.dtype,shape = (width,))
    start_have = 0
    stop_have = arr.size
    if stop_want < start_have or stop_have < start_want:
        return out
    if start_want < start_have <= stop_want <= stop_have:
        nzeros = start_have - start_want # zero-pad at beginning of output array
        samples_present = width - nzeros 
        out[nzeros : nzeros + samples_present] = arr[0:samples_present]
        return out
    if start_have <= start_want <= stop_have < stop_want:
        nzeros = stop_want - stop_have # zero-pad at end of output array
        out[0 : width - nzeros] = arr[start_want:stop_have]
        return out
    if start_want <= start_have < stop_have <= stop_want:
        nzeros = start_have - start_want
        samples_present = stop_have - start_have
        out[nzeros : nzeros + samples_present] = arr
        return out
    else:
        return arr[start_want:stop_want]

def get_aligned_scans(
    bbdata_a: BBData, 
    bbdata_b: BBData, 
    t_a_index: np.ndarray, 
    wij: int, 
    tau: np.ndarray, 
    complex_conjugate_convention: int=-1, 
    intra_channel_sign: int=1, 
    sample_rate: float =2.56,
    npointing:int=0,
    n_pol:int=2,
    zp:bool=True
    ) -> Tuple[np.ndarray,np.ndarray]:
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

    Technical remarks on delay compensation:
    On floor vs round: it doesn't matter AS LONG AS you do a sub-sample rotation (or better yet, a frac samp correction)! Suppose your total delay is 10.6 frames.
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
    if len(np.unique(t_a_index))==1 and not zp:
        aligned_a[:, ...] = bbdata_a['tiedbeam_baseband'][:,npointing:npointing+n_pol,
                                                          t_a_index[0]:t_a_index[0] + wij]

    elif len(np.unique(t_a_index)) > 1 and not zp:
        for i in range(len(t_a_index)):
            aligned_a[i, ...] = bbdata_a['tiedbeam_baseband'][i,npointing:npointing+n_pol,
                                                            t_a_index[i]:t_a_index[i] + wij]
    elif zp:
        for i in range(len(t_a_index)):
            for j in range(n_pol):
                aligned_a[i,j,:] = getitem_zp1d(bbdata_a['tiedbeam_baseband'][i,j,:],t_a_index[i],t_a_index[i] + wij)
    # aligned_a = bbdata_a['tiedbeam_baseband'][freq_id,...,t_a_index:t_a_index + wij]
    # initialize aligned B array
    aligned_b = np.zeros_like(aligned_a)
        
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

    if not zp:
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
            logging.warning("based on specified start time and scan length, over half the data is missing from telescope XX.")

        for i in range(len(pad_index_b)):
            aligned_b[i, ..., pad_index_b[i]:pad_index_b[i]+new_wij[i]] = \
                bbdata_b['tiedbeam_baseband'][i, ...,start_index_we_have_at_b[i]:start_index_we_have_at_b[i]+new_wij[i]] * correction_factor[i]
        # multiply by the correction factor to ensure that a continuum source, when correlated, has the correct flux corresponding to the desired w_ij, even when we run out of data.
        aligned_b = aligned_b[..., :wij]
    elif zp:
        for i in range(len(t_a_index)):
            for j in range(n_pol):
                aligned_b[i,j,:] = getitem_zp1d(bbdata_b['tiedbeam_baseband'][i,j,:],start_index_we_want_at_b[i],start_index_we_want_at_b[i] + wij)

    time_we_have_at_b = (delta_A_B+int_delay*sample_rate*1e-6)  # s
    sub_frame_tau = np.array([tau[i, :wij] - time_b*1e6 for time_b, i in zip(
        time_we_have_at_b, range(len(tau)))])  # sub-frame delay at start time in mircoseconds

    aligned_b = frac_samp_shift(aligned_b,
                                           f0=bbdata_b.index_map["freq"]["centre"][:],
                                           sub_frame_tau=sub_frame_tau,
                                           complex_conjugate_convention=complex_conjugate_convention,
                                           intra_channel_sign=intra_channel_sign,
                                           sample_rate=sample_rate)
    return aligned_a, aligned_b

#### faster option is with gpus
def frac_samp_shift(
    data:np.ndarray, 
    f0:np.ndarray, 
    sub_frame_tau:np.ndarray, 
    complex_conjugate_convention:int=-1, 
    intra_channel_sign:int=1, 
    sample_rate: float=2.56,
    max_frames: int=MAX_FRAC_SAMP_LENGTH, 
    ) -> np.ndarray:
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

    Outputs:
    -------
    np.ndarray of shape (nfreq,npol*npointing,ntime)
    
    """
    # glorified element wise multiplication
    #data will be of shape (nfreq,npol,ntime)
    n = data.shape[-1]
    if n<=max_frames:        
        f = np.fft.fftfreq(n, sample_rate)
        # transfer_func is now of shape (nfreq,ntime)
        transfer_func = np.exp(intra_channel_sign*2j * np.pi * f[np.newaxis,:] * np.median(sub_frame_tau,axis=-1)[:,np.newaxis])  # apply dphi/dfreq
        frac_samp= np.fft.ifft(
                np.fft.fft(data, axis=-1) * transfer_func[:,np.newaxis,], axis=-1
            ) * (np.exp(complex_conjugate_convention*2j * np.pi * f0[:,np.newaxis] * sub_frame_tau))[:, np.newaxis, :]   # apply phi'
        return frac_samp
    else:
        n1=n//2
        data_chunk1=data[:,:,:n1]
        data_chunk2=data[:,:,n1:]
        sub_frame_tau1=sub_frame_tau[:,:n1]
        sub_frame_tau2=sub_frame_tau[:,n1:]
        fft_corrected_chunk1=frac_samp_shift(
            data=data_chunk1,
            f0=f0,
            sub_frame_tau=sub_frame_tau1,
            complex_conjugate_convention=complex_conjugate_convention,
            intra_channel_sign=intra_channel_sign,
            sample_rate=sample_rate,
            max_frames=max_frames
            )
        fft_corrected_chunk2=frac_samp_shift(
            data=data_chunk2,
            f0=f0,
            sub_frame_tau=sub_frame_tau2,
            complex_conjugate_convention=complex_conjugate_convention,
            intra_channel_sign=intra_channel_sign,
            sample_rate=sample_rate,
            max_frames=max_frames
            )
        return np.append(fft_corrected_chunk1,fft_corrected_chunk2,axis=-1)



def intrachannel_dedisp(
    data: np.ndarray,
    DM: float,
    f0: np.ndarray,
    sample_rate: float = 2.56
    ) -> np.ndarray:
    """Intrachannel dedispersion: brings data to center of channel.
    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook, but ONLY the last term (proportional to f^2), not the other two terms (independent of f and linearly proportional to f respectively).
    
    Inputs:
    -------
    data : np.ndarray of shape (nfreq,npol*npointing,ntime)
    f0 : np.ndarray of shape (nfreq) holding channel centers.
    sample_rate : sampling rate of data in microseconds
    
    Outputs:
    -------
    np.ndarray of shape (nfreq,npol*npointing,ntime)
    """
    if DM==0: #save computation time
        return data
    else:        
        n = data.shape[-1]
        f = np.fft.fftfreq(n,d = sample_rate)
        transfer_func = np.exp(2j * np.pi * K_DM * DM * 1e6 * f[np.newaxis,:]**2 / f0[:,np.newaxis]**2 / (f[np.newaxis,:] + f0[:,np.newaxis]))  
        return np.fft.ifft(np.fft.fft(data, axis=-1) * transfer_func[:,np.newaxis,:],axis=-1)