"""
Fringestops all input *stations* and cross correlates baseband data over *all* N(N-1)/2 baselines
The "core" module that should be called by the "outer layer" corr_job.py
All bugs are the responsibility of Shion Andrew   
"""

import collections
import logging
import time
from decimal import Decimal
from typing import List, Optional, Tuple, Union

import astropy.units as un
import numpy as np
from astropy.time import Time, TimeDelta
from baseband_analysis.core.bbdata import BBData
from pycalc11 import Calc

from pyfx.fft_corr import (
    basic_correlator as basic_correlator,
)  # could swap out correlator here

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
MAX_FRAC_SAMP_LENGTH = 32187  # maximum FFT length, chosen to keep delay rate drift (on Earth) within 1/10th of a frame

def get_pointing_map(pointing_spec,tb_locations,mode = '1to1',tolerance_arcsec = 300,ties_ok = False):
    """Assign pointings to beam indices.

    Parameters
    ----------
    pointing_spec : np.ndarray of shape (npointing,)

    tb_locations : np.ndarray of shape (nbeam,) where nbeam is 2x the number of distinct locations (i.e. dual pol beams)

    Returns
    -------
    beam_indices : list of ints
        For N pointings, this will return a list of length 2N.
        If mode = '1to1', then for N pointings it will return [0,1,2,3,...], 
        and assume that the jth beamformer pointing is the 2jth beam, and that the 2j+1th beam is the other polarization.

        If mode = 'nearest':
        It will take the nearest beam within tolerance_arcsecond.
    """

    assert (np.diff(tb_locations['ra'][:])[::2] == 0).all(), "Polarizations not repeated?" # check repeated pols
    assert (np.diff(tb_locations['dec'][:])[::2] == 0).all(), "Polarizations not repeated?" # check repeated pols
    tbl = tb_locations[::2].copy()
    beam_indices = []    
    if mode == '1to1':
        assert len(pointing_spec) == len(tbl), "different numbers of pointings and tied beam locations?"
        beam_indices = np.arange(len(pointing_spec))
    if mode == 'nearest':
        from astropy.coordinates import SkyCoord
        beam_sc = SkyCoord(tb_locations['ra'][::2],tb_locations['dec'][::2],unit = 'deg')
        for iipointing, pointing in enumerate(pointing_spec):
            seps_this_pointing = SkyCoord(ra = pointing['corr_ra'],dec = pointing['corr_dec'],unit = 'deg').separation(beam_sc).to_value('arcsec')
            min_dist = min(seps_this_pointing)
            assert min_dist < tolerance_arcsec
            if not ties_ok:
                assert np.sum(seps_this_pointing  == min_dist) == 1, "Two exactly equidistant pointings! Not allowed."
            beam_indices.append(2 * np.argmin(min_dist))
    return beam_indices

def autocorr_core(
    pointing_spec : np.ndarray,
    assign_pointing : str,
    DM: float,
    bbdata_a: BBData,
    t_a: np.ndarray,
    window: np.ndarray,
    R: np.ndarray,
    max_lag: int,
    n_pol: int = 2,
) -> np.ndarray:
    """Auto-correlates data and downselects over lag.

    Parameters
    ----------
    DM : float
        The DM with which the zeroth pointing of the data is de-smeared before the final gating. for continuum sources, set dispersion measure to 0.

    bbdata_a : BBData object
        At bare minimum, needs to have "tiedbeam_baseband" data of size (nfreq, nbeam, ntime).

    t_a : np.ndarray of int of shape (nfreq, npointing, nscan).
        start index of the integration, relative to bbdata_a['time0']['ctime'] in units of 2.56 microsec, as a function of frequency channels, pointing index, and time in units of :window: (i.e. scan number).

    window : np.ndarray of int of shape (npointing, nscan).
        duration of the scan, in units of 2.56 microsec, as a function of pointing and time (i.e. scan number).

    R : np.ndarray of float of shape (nfreq, npointing, nscan).
        Fraction R <= 1 of the scan, that gets down-selected before integration. In other words, we integrate between t_a + window // 2 +- r * window / 2

    max_lag : int
        maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data). TODO: Outer layer of the code should check that this is less than 1/2 of the window size times R.
        set this to 20 for a good balance between space efficiency and good noise statistics.

    n_pol : int
        number of polarizations in data -- always 2.

    Returns
    -------
    auto_vis : np.array of floats of shape (nfreq, npointing, npol, npol, 2 * nlag + 1, nscan)
        Array of autocorrelation data.
    """
    n_freq = bbdata_a.nfreq
    n_scan = np.size(t_a, axis=-1)
    n_pointings = len(pointing_spec)

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, 2 * max_lag + 1, n_scan)
    auto_vis = np.zeros(vis_shape, dtype=bbdata_a["tiedbeam_baseband"].dtype)
    f0 = bbdata_a.index_map["freq"]["centre"]  # shape is (nfreq)
    beam_indices = get_pointing_map(
            tb_locations = bbdata_a['tiedbeam_locations'][:],
            pointing_spec = pointing_spec,
            mode = assign_pointing)
    for kkpointing in range(n_pointings):
        beam_idx = beam_indices[kkpointing] # assign a beam to the pointing
        for jjscan in range(n_scan):
            logging.info(f'Pointing {kkpointing+1}/{n_pointings}; Scan {jjscan+1}/{n_scan}')
            wij = int(window[kkpointing, jjscan])
            t_a_indices = t_a[:, kkpointing, jjscan]  # array of length 1024
            ## clip telescope A data ##
            a_shape = (bbdata_a["tiedbeam_baseband"].shape[0],2,wij)
            clipped_a = np.zeros(a_shape, dtype=bbdata_a["tiedbeam_baseband"].dtype) # clipped_a.shape = (n_freq, n_pointing * 2, w_ij)
            for i in range(len(t_a_indices)): # loop over frequency
                for j in range(n_pol): # loop over pols
                    clipped_a[i, j, :] = getitem_zp1d(
                        bbdata_a["tiedbeam_baseband"][i, beam_idx + j],
                        t_a_indices[i],
                        t_a_indices[i] + wij,
                    )

            ######### intrachannel de-dispersion, vectorized over freq ###
            scan_a_fs_cd = intrachannel_dedisp(clipped_a[:,beam_idx : beam_idx + 2,:], DM[kkpointing], f0=f0)
            r_jjscan = R[:, kkpointing, jjscan] # np array of size (nfreq)

            #### FFT correlation, vectorized over freq if possible ####
            if len(np.unique(r_jjscan)) == 1: # uniform freq resolution; vectorize!
                r_ij = r_jjscan[0]
                start = int((wij - wij * r_ij) // 2)
                stop = int((wij + wij * r_ij) // 2)
                #######################################################
                ########## auto-correlate the on-signal ##############
                for pol_0 in range(n_pol):
                    for pol_1 in range(n_pol):
                        auto_vis[:, kkpointing, pol_0, pol_1, :, jjscan] = (
                            basic_correlator(
                                scan_a_fs_cd[:, pol_0, start:stop],
                                scan_a_fs_cd[:, pol_1, start:stop],
                                max_lag=max_lag,
                            )
                        )
                        """auto_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                            (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)"""
            else: # go freq by freq
                for iifreq, r_ij in enumerate(r_jjscan): # loop over frequency
                    start = int((wij - wij * r_ij) // 2)
                    stop = int((wij + wij * r_ij) // 2)
                    #######################################################
                    ########## auto-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            if pol_0 == pol_1:
                                auto_vis[iifreq, kkpointing, pol_0, pol_1, :, jjscan] = (
                                    basic_correlator(
                                        scan_a_fs_cd[iifreq, pol_0, start:stop],
                                        scan_a_fs_cd[iifreq, pol_1, start:stop],
                                        max_lag=max_lag,
                                    )
                                )
                                """auto_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                                    (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)"""

    return auto_vis


def get_start_times(
    bbdata_top: BBData,  # station at which toa is measured
    sample_rate: float = 2.56,  # microseconds
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unix times from bbdata_top
    Parameters
    ----------
    bbdata_top - bbdata where topocentric time is defined (usually chime)

    Outputs
    ----------
    t0_a, t0_a_offset - np.ndarray
        Recorded unix timestamps (ctime and ctime_offset)
    """
    t0_a = bbdata_top["time0"]["ctime"][:]
    t0_a_offset = bbdata_top["time0"]["ctime_offset"][:]
    return t0_a, t0_a_offset


def get_delays(
    telescope_index: int,
    ref_start_time: np.ndarray,
    ref_start_time_offset: np.ndarray,
    wij: int,
    pycalc_results: Calc,
    ref_frame: int,
    sample_rate: float = 2.56,
    pointing = None,
):
    """Get delays towards each pointing.
    
    Parameters
    ----------
    telescope_index : int
        index of telescope to be fringestopped
    
    ref_start_time : np.ndarray of float64 of shape (n_freq)
        topocentric unix time at which geodelays should be evaluated
    
    ref_start_time_offset : np.ndarray of float64 of shape (n_freq)
        a second float to hold topocentric unix time to high precision
    
    ref_frame : int
        index corresponding to station where topocentric unix time is defined (usually CHIME)

    pointing : int
        index of the pointing for which we want delays.
        N.b. currently, all pointings are calculated but only of the pointings is kept.
        This is really inefficient, since currently this is called in a for loop over all pointings.
        Optimize later.

    Returns
    -------
    geodelay : np.ndarray of float64 of shape (nfreq, n_pointing, duration_frames)
        Geometric delay in microseconds for the requested phase center(s).
        N.b. technically
    """
    if ref_frame == telescope_index:
        return np.zeros((ref_start_time.shape[0], wij))  # save time
    delta_ctime = ref_start_time - ref_start_time[0]
    delta_ctime_offset = ref_start_time_offset - ref_start_time_offset[0]
    delta_t = delta_ctime + delta_ctime_offset
    dt_vals = (
        sample_rate * 1e-6 * np.arange(wij) + delta_t[:, np.newaxis]
    )  # nfreq,nframe
    ref_start_time = Time(
        ref_start_time[0],
        val2=ref_start_time_offset[0],
        format="unix",
        precision=9,
    )
    dt_vals0 = (ref_start_time - pycalc_results.times[0]).sec  # should always be <1s.
    delays_flattened = pycalc_results.delays_dt(dt_vals0 + dt_vals.flatten())
    # if ref_frame is None: #delays are relative to the geocenter
    #    geodelays_flattened=delays_flattened[:,0,telescope_index,:]
    # else:
    geodelays_flattened = (
        delays_flattened[:, 0, telescope_index, pointing]
        - delays_flattened[:, 0, ref_frame, pointing]
    )  # units of seconds
    geodelays = (
        geodelays_flattened.reshape(dt_vals.shape) * 1e6
    )  # microseconds #nfreq,nframe
    return geodelays


def fringestop_station(
    bbdata: BBData,
    bbdata_top: BBData,
    pointing_spec : np.ndarray,
    assign_pointing : str,
    t_a: np.ndarray,
    window: np.ndarray,
    pycalc_results: Calc,
    station_index: int,
    ref_frame: int,
    sample_rate: float = 2.56,
    n_pol: int = 2,
    complex_conjugate_convention: int = -1,
    intra_channel_sign: int = 1,
    weight: Optional[np.ndarray] = None,
    max_frames: int = MAX_FRAC_SAMP_LENGTH,
) -> np.ndarray:
    """Returns fringestopped scans of data for a station of index station_index.

    bbdata : BBData
        The BBData out of which we extract fringestopped scans.

    bbdata_top : BBData
        The reference BBData defining the topocentric frame to which we align things.

    t_a : np.ndarray of int
        Integer frames for BBData at the topocentric reference.
    
    window : np.ndarray of int of shape (1024, n_pointing, n_scan)
        Duration of scans
    
    station_index : int
        The index of the station to be fringestopped, to be used by pycalc.
    
    ref_frame :
        The index of the reference station.
    
    sample_rate : float
        In microseconds (2.56 default)

    n_pol : int
        Should be 2
    
    complex_conjugate_convention : -1
        Sign convention for fringe rotation
    intra_channel_sign : 1
        Sign convention for fractional sample correction
    
    assign_pointing : str
        Either '1to1' or 'nearest': See documentation of get_pointing_map() 

    """
    logging.info(f'Fringestopping station {station_index}')
    n_freq = len(bbdata_top.freq)
    n_scan = np.size(t_a, axis=-1)
    n_pointings = len(pointing_spec)
    n_freq_B = len(bbdata.freq)
    assert (
        n_freq_B == n_freq
    ), f"There appear to be {n_freq} frequency channels in telescope A and {n_freq_B} frequency channels in telescope B. Please pass in these bbdata objects with frequency channels aligned (i.e. nth index along the frequency axis should correspond to the *same* channel in telescope A and B)"
    bbdata_shape = (n_freq, 2 * n_pointings, n_scan, max(window.flatten()))
    fringestopped_data = np.zeros(
        bbdata_shape, dtype=bbdata["tiedbeam_baseband"].dtype
    )  # zeropadded on edges of wij
    f0 = bbdata.index_map["freq"]["centre"]  # shape is (nfreq)
    beam_indices = get_pointing_map(
        tb_locations = bbdata['tiedbeam_locations'][:],
        pointing_spec = pointing_spec,
        mode = assign_pointing)
    for kkpointing in range(n_pointings):
        beam_idx = beam_indices[kkpointing] # assign a beam to the pointing
        for jjscan in range(n_scan):
            wij = window[kkpointing, jjscan]
            t_a_indices = t_a[:, kkpointing, jjscan]  # array of length 1024

            ref_ctime, ref_ctime_offset = get_start_times(
                bbdata_top=bbdata_top, sample_rate=sample_rate
            )

            ref_start_time_offset = ref_ctime_offset + t_a_indices * (
                sample_rate * 1e-6
            )  # array of length 1024

            geodelays = get_delays(
                ref_frame=ref_frame,
                telescope_index=station_index,
                ref_start_time=ref_ctime,
                ref_start_time_offset=ref_start_time_offset,
                wij=wij,
                pycalc_results=pycalc_results,
                sample_rate=sample_rate,
                pointing=kkpointing # inefficient but whatever I hope pycalc is fast enough
            )
            scan_fs = fringestop_scan(
                bbdata,
                ref_ctime=ref_ctime,
                ref_ctime_offset=ref_ctime_offset,
                t_a_index=t_a_indices,
                wij=wij,
                geodelays=geodelays,
                complex_conjugate_convention=complex_conjugate_convention,
                intra_channel_sign=intra_channel_sign,
                beam_idx = 2 * beam_idx, # factor of two because 2 pols
                max_frames=max_frames,
            )
            fringestopped_data[:,
                               2 * kkpointing : 2 * kkpointing + 2, 
                               jjscan,
                               :wij] = scan_fs
    return fringestopped_data

def crosscorr_core(
    bbdata_a_fs: np.ndarray,
    bbdata_b_fs: np.ndarray,
    f0: np.ndarray,
    window: np.ndarray,
    R: np.ndarray,
    DM: float,
    index_A: int,
    index_B: int,
    max_lag: int,
    ref_frame: int,
    sample_rate: float = 2.56,
    n_pol: int = 2,
    complex_conjugate_convention: int = -1,
    intra_channel_sign: int = 1,
    weight: Optional[np.ndarray] = None,
    max_frames: int = MAX_FRAC_SAMP_LENGTH,
) -> np.ndarray:
    """Fringestops, coherently dedisperses, and cross correlates data
    Parameters
    ----------
    bbdata_a_fs : np.ndarray of shape (n_freq, 2 * n_pointing, n_scan, window_size)
        At bare minimum, needs to have "tiedbeam_baseband" data of size (nfreq, npointing*npol, ntime).

    bbdata_b_fs : np.ndarray of shape (n_freq, 2 * n_pointing, n_scan, window_size)
        telescope B baseband data. Data must have matching index_map['freq'] as bbdata_a. index_map['freq']['centre'] must also be in MHz.
    f0 : np.ndarray of central channel frequencies -> intrachannel_dedisp

    window : np.ndarray of ints of shape (n_freq, n_pointing, n_time)
        duration of the scan, in units of 2.56 microsec, as a function of frequency & pointing and time (i.e. scan number).

    R : np.ndarray of float of shape (nfreq, npointing, nscan).
        Fraction R <= 1 of the scan, that gets down-selected before integration. In other words, we integrate between t_a + window // 2 +- r * window / 2

    DM : np.ndarray of float of shape (npointing,)
        The DM with which the zeroth pointing of the data is de-smeared before the final gating. for continuum sources, set dispersion measure to 0.

    index_A :
        where telescope A corresponds to in pycalc_results.

    index_B :
        where telescope B corresponds to in pycalc_results.

    sample_rate :
        rate at which data is sampled in microseconds

    max_lag : int
        maximum (absolute value) lag (in frames) for auto-correlation (useful for very long time series data). TODO: Outer layer of the code should check that this is less than 1/2 of the window size times R.
        set this to 20 for a good balance between space efficiency and good noise statistics.

    ref_frame : int
        Index (e.g. index_A, index_B) corresponding to reference frame in which data will be fringestopped to. 0 corresponds to the geocenter. If None, will default to index_A.

    n_pol : int
        number of polarizations in data -- always 2.

    pycalc_results :
        pycalc11 Calc object, which is used to calculate geometric delays. Calc object should be initialized outside of this function and driver should have already been run (i.e. ci.run_driver())

    complex conjugate convention :
        should be a value of -1 if the baseband data is complex conjugated with respect to the sky, 1 otherwise

    intra_channel_sign :
        a sign to account for a reflection of frequencies about zero (e.g. in iq/baseband data). Should be -1 if frequencies within a channel are reflected about 0, 1 otherwise.

    fast :
        [not yet implemented] If False, check that all sums of times (to nanosecond precision) are less than 64 bytes
        If True, trust the process.

    weight :
        array of shape (nfreq, nscan, npointings,ntime) that specifies what weighting to apply to the data **relative to the start time given by t_a**.
        The shape of weight[:,jjscan,kkpointing] should be window[jjscan,kkpointing]

    Outputs:
    -------
    cross_vis :
        array of cross_vis correlation visibilities with shape (nfreq, npointing, npol, npol, nlag,nscan)
    """
    n_freq = bbdata_a_fs.shape[0]
    n_scan = np.size(R, axis=-1)
    n_pointings = window.shape[0]
    # SA: basing this off of how the data is arranged now, may want to change
    n_freq_B = bbdata_b_fs.shape[0]
    assert (
        n_freq_B == n_freq
    ), f"There appear to be {n_freq} frequency channels in telescope A and {n_freq_B} frequency channels in telescope B. Please pass in these bbdata objects with frequency channels aligned (i.e. nth index along the frequency axis should correspond to the *same* channel in telescope A and B)"

    vis_shape = (n_freq, n_pointings, n_pol, n_pol, 2 * max_lag + 1, n_scan)
    cross_vis = np.zeros(vis_shape, dtype=bbdata_a_fs.dtype)

    for kkpointing in range(n_pointings):
        for jjscan in range(n_scan):
            logging.info(f'Pointing {kkpointing+1}/{n_pointings}; Scan {jjscan+1}/{n_scan}')
            wij = window[kkpointing, jjscan]
            scan_a_fs = bbdata_a_fs[
                :, kkpointing * n_pol : kkpointing * n_pol + n_pol, jjscan,:wij
            ]
            scan_b_fs = bbdata_b_fs[
                :, kkpointing * n_pol : kkpointing * n_pol + n_pol, jjscan,:wij
            ]
            #######################################################
            ######### intrachannel de-dispersion ##################
            scan_a_fs_cd = intrachannel_dedisp(scan_a_fs, DM[kkpointing], f0=f0)
            scan_b_fs_cd = intrachannel_dedisp(scan_b_fs, DM[kkpointing], f0=f0)

            ## weight the data
            if type(weight) == np.ndarray:
                scan_a_fs_cd *= weight[:, np.newaxis, jjscan, kkpointing, :]
                scan_b_fs_cd *= weight[:, np.newaxis, jjscan, kkpointing, :]

            #######################################################
            # Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on-signal ######
            r_jjscan = R[:, kkpointing, jjscan]  # np array of size (nfreq)

            if len(np.unique(r_jjscan)) == 1:
                # we can easily vectorize over frequency
                r_ij = r_jjscan[0]
                start = int((wij - wij * r_ij) // 2)
                stop = int((wij + wij * r_ij) // 2)
                if start == stop:
                    logging.warning(
                        "r_ij includes less than 1 frame: visibilities will be set to 0"
                    )
                else:
                    #######################################################
                    ########## cross-correlate the on-signal ##############
                    for pol_0 in range(n_pol):
                        for pol_1 in range(n_pol):
                            assert not np.isnan(
                                np.min(scan_a_fs_cd[:, pol_0, start:stop].flatten())
                            ), "Scan parameters have been poorly defined for telescope A. Please ensure there are no nans in the baseband data"
                            assert not np.isnan(
                                np.min(scan_b_fs_cd[:, pol_0, start:stop].flatten())
                            ), "Scan parameters have been poorly defined for telescope B. Please ensure there are no nans in the baseband data"
                            cross_vis[:, kkpointing, pol_0, pol_1, :, jjscan] = (
                                basic_correlator(
                                    scan_a_fs_cd[:, pol_0, start:stop],
                                    scan_b_fs_cd[:, pol_1, start:stop],
                                    max_lag=max_lag,
                                )
                            )
                            # cross_vis[:, kkpointing, pol_0, pol_1,:,jjscan] = np.concatenate(
                            #    (_vis[:,:max_lag+1], _vis[:,-max_lag:]),axis=-1)
            else:
                # loop over frequency channel
                for freq in range(len(r_jjscan)):
                    r_ij = r_jjscan[freq]
                    start = int((wij - wij * r_ij) // 2)
                    stop = int((wij + wij * r_ij) // 2)
                    if start == stop:
                        logging.warning(
                            "r_ij includes less than 1 frame: visibilities for this channel will be set to 0"
                        )
                    else:
                        #######################################################
                        ########## cross-correlate the on-signal ##############
                        for pol_0 in range(n_pol):
                            for pol_1 in range(n_pol):
                                assert not np.isnan(
                                    np.min(
                                        scan_a_fs_cd[freq, pol_0, start:stop].flatten()
                                    )
                                ), "Scan parameters have been poorly defined for telescope A. Please ensure there are no nans in the baseband data"
                                assert not np.isnan(
                                    np.min(
                                        scan_b_fs_cd[freq, pol_0, start:stop].flatten()
                                    )
                                ), "Scan parameters have been poorly defined for telescope B. Please ensure there are no nans in the baseband data"
                                cross_vis[freq, kkpointing, pol_0, pol_1, :, jjscan] = (
                                    basic_correlator(
                                        scan_a_fs_cd[freq, pol_0, start:stop],
                                        scan_b_fs_cd[freq, pol_1, start:stop],
                                        max_lag=max_lag,
                                    )
                                )
                                # cross_vis[freq, kkpointing, pol_0, pol_1, :,jjscan] = np.concatenate(
                                #    (_vis[:max_lag+1], _vis[-max_lag:]),axis=-1)
    return cross_vis


def getitem_zp1d(arr, start_want, stop_want):
    """Slices a 1d np.array arr[start_want:stop_want] but assumes start is strictly less than stop.

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

    Returns
    -------
    out : 1d np.ndarray of dtype arr.dtype of size stop_want - start_want.
    """
    width = int(stop_want - start_want)
    assert width >= 0, "Negative scan length not allowed; check your w_ij"
    out = np.zeros(dtype=arr.dtype, shape=(width,))
    start_have = 0
    stop_have = arr.size
    if stop_want < start_have or stop_have < start_want:
        return out
    if start_want < start_have <= stop_want <= stop_have:
        nzeros = start_have - start_want  # zero-pad at beginning of output array
        samples_present = width - nzeros
        out[nzeros : nzeros + samples_present] = arr[0:samples_present]
        return out
    if start_have <= start_want <= stop_have < stop_want:
        nzeros = stop_want - stop_have  # zero-pad at end of output array
        out[0 : width - nzeros] = arr[start_want:stop_have]
        return out
    if start_want <= start_have < stop_have <= stop_want:
        nzeros = start_have - start_want
        samples_present = stop_have - start_have
        out[nzeros : nzeros + samples_present] = arr
        return out
    else:
        return arr[start_want:stop_want]


#### faster option is with gpus
def frac_samp_shift(
    data: np.ndarray,
    f0: np.ndarray,
    sub_frame_tau: np.ndarray,
    complex_conjugate_convention: int = -1,
    intra_channel_sign: int = 1,
    sample_rate: float = 2.56,
    max_frames: int = MAX_FRAC_SAMP_LENGTH,
) -> np.ndarray:
    """
    Coherently shifts data within a channel via a fractional-sample time shift...
    ...whose phase versus frequency is of the form exp(2j*pi*f*sub_frame_tau),
    where f is the intra-channel frequency.

    For 
    If the data has ntime > max_frames, the data will be broken in half and this will be recursively called.
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
    # data will be of shape (nfreq,npol,ntime)
    n = data.shape[-1]
    if n <= max_frames:
        f = np.fft.fftfreq(n, sample_rate)
        # transfer_func is now of shape (nfreq,ntime)
        transfer_func = np.exp(
            intra_channel_sign
            * 2j
            * np.pi
            * f[np.newaxis, :]
            * np.median(sub_frame_tau, axis=-1)[:, np.newaxis]
        )  # apply dphi/dfreq
        frac_samp = (
            np.fft.ifft(
                np.fft.fft(data, axis=-1)
                * transfer_func[
                    :,
                    np.newaxis,
                ],
                axis=-1,
            )
            * (
                np.exp(
                    complex_conjugate_convention
                    * 2j
                    * np.pi
                    * f0[:, np.newaxis]
                    * sub_frame_tau
                )
            )[:, np.newaxis, :]
        )  # apply phi'
        return frac_samp
    else:
        n1 = n // 2
        data_chunk1 = data[:, :, :n1]
        data_chunk2 = data[:, :, n1:]
        sub_frame_tau1 = sub_frame_tau[:, :n1]
        sub_frame_tau2 = sub_frame_tau[:, n1:]
        fft_corrected_chunk1 = frac_samp_shift(
            data=data_chunk1,
            f0=f0,
            sub_frame_tau=sub_frame_tau1,
            complex_conjugate_convention=complex_conjugate_convention,
            intra_channel_sign=intra_channel_sign,
            sample_rate=sample_rate,
            max_frames=max_frames,
        )
        fft_corrected_chunk2 = frac_samp_shift(
            data=data_chunk2,
            f0=f0,
            sub_frame_tau=sub_frame_tau2,
            complex_conjugate_convention=complex_conjugate_convention,
            intra_channel_sign=intra_channel_sign,
            sample_rate=sample_rate,
            max_frames=max_frames,
        )
        return np.append(fft_corrected_chunk1, fft_corrected_chunk2, axis=-1)


def intrachannel_dedisp(
    data: np.ndarray, DM: float, f0: np.ndarray, sample_rate: float = 2.56
) -> np.ndarray:
    """Intrachannel dedispersion: brings data to center of channel.
    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook.
    Following the discussion in Leung 2024, ONLY the last term (proportional to f^2), 
    not the other two terms (independent of f and linearly proportional to f respectively).

    Inputs:
    -------
    data : np.ndarray of shape (nfreq,npol*npointing,ntime)
    f0 : np.ndarray of shape (nfreq) holding channel centers.
    sample_rate : sampling rate of data in microseconds

    Outputs:
    -------
    np.ndarray of shape (nfreq,npol*npointing,ntime)
    """
    if DM == 0:  # save computation time
        return data
    else:
        n = data.shape[-1]
        f = np.fft.fftfreq(n, d=sample_rate)
        transfer_func = np.exp(
            2j
            * np.pi
            * K_DM
            * DM
            * 1e6
            * f[np.newaxis, :] ** 2
            / f0[:, np.newaxis] ** 2
            / (f[np.newaxis, :] + f0[:, np.newaxis])
        )
        return np.fft.ifft(
            np.fft.fft(data, axis=-1) * transfer_func[:, np.newaxis, :], axis=-1
        )


def fringestop_scan(
    bbdata: BBData,
    ref_ctime: np.ndarray,
    ref_ctime_offset: np.ndarray,
    t_a_index: np.ndarray,
    wij: int,
    geodelays: np.ndarray,
    complex_conjugate_convention: int = -1,
    intra_channel_sign: int = 1,
    sample_rate: float = 2.56,
    beam_idx : int = 0,
    n_pol: int = 2,
    max_frames: int = MAX_FRAC_SAMP_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.
    Inputs:
    -------
    bbdata : BBData
        A BBData object, with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    toa : np.array of shape (1024)
        An array of start times in the geocenter frame

    w_ij : int
        window length. Should be an integer, and brownie points for a good FFT length.

    geodelays : np.array (nfreq, n_frame) of dtype np.float
        A delay in microseconds to apply to BBData_b, corresponding to the geometric delay.
        The first index is the delay evaluated at time t_ij_a

    beam_idx : int
        To get the Nth (dual-pol) beam in a BBData, pass in beam_idx which gets indexed into BBData as data['tiedbeam_baseband'][:,beam_idx : beam_idx + 2,:]
    Outputs:
    -------
    aligned_bbdata : np.array
        A dual-pol scan of shape (2,w_ij)
    """

    relative_start_time_we_want = (
        geodelays[:, 0] * 1e-6
    )  # start times in s relative to ref_start_time
    fringestopped_shape = list(
        bbdata["tiedbeam_baseband"][:, beam_idx : beam_idx + n_pol, :].shape
    )
    fringestopped_shape[-1] = wij  # (nfreq,npol,wij)
    aligned_bbdata = np.zeros(
        fringestopped_shape, dtype=bbdata["tiedbeam_baseband"].dtype
    )

    # calculate the additional offset between A and B in the event that the (samples points of) A and B are misaligned in absolute time by < 1 frame
    # i.e. to correctly fringestop, we must also account for a case such as:
    ## A:    |----|----|----|----|----| ##
    ## B: |----|----|----|----|----|    ##

    # find out the index corresponding to the frame number closest in time to ref_start_time
    desired_Time = Time(ref_ctime, val2=ref_ctime_offset, format="unix", precision=9)
    tel_Time = Time(
        bbdata["time0"]["ctime"][:],
        val2=bbdata["time0"]["ctime_offset"][:],
        format="unix",
        precision=9,
    ) # tel_Time.shape = (n_freq,)
    delta_A_B = (tel_Time - desired_Time).to_value("sec")
    int_delay = np.array(
        [
            int(np.round((rel_time - frame_offset) / (sample_rate * 1e-6)))
            for rel_time, frame_offset, in zip(relative_start_time_we_want, delta_A_B)
        ]
    )
    # frame number closest to start time
    start_index_we_want_at_b = int_delay + t_a_index

    for i in range(len(start_index_we_want_at_b)):
        for j in range(n_pol):
            aligned_bbdata[i, j, :] = getitem_zp1d(
                bbdata["tiedbeam_baseband"][i, j, :],
                start_index_we_want_at_b[i],
                start_index_we_want_at_b[i] + wij,
            )

    if np.nanmax(geodelays) != 0:  # save time if input is bbdataA
        time_we_have_at_b = delta_A_B + int_delay * sample_rate * 1e-6  # s
        sub_frame_tau = np.array(
            [
                geodelays[i, :wij] - time_b * 1e6
                for time_b, i in zip(time_we_have_at_b, range(len(geodelays)))
            ]
        )  # sub-frame delay at start time in mircoseconds
        aligned_bbdata = frac_samp_shift(
            aligned_bbdata,
            f0=bbdata.index_map["freq"]["centre"][:],
            sub_frame_tau=sub_frame_tau,
            complex_conjugate_convention=complex_conjugate_convention,
            intra_channel_sign=intra_channel_sign,
            max_frames=max_frames,
            sample_rate=sample_rate,
        )

    return aligned_bbdata

def fs_ref2center_vec(scan,integer_shift_ref,phase_shift_ref,add=False):
    """ Fringestops a long, potentially non-contiguous (i.e. gated) scan of baseband data, via vectorized shifting.
    Note that Calvin has a Cython version of this exists; this is just for development for now.
    scan : np.array
        1d array of complex baseband data.
    integer_shift_ref : np.array
        1d array of integers, of the same shape as scan.
    phase_shift_ref : np.array
        1d array of integers, of the same shape as scan.
    """
    
    scan_shifted = np.zeros_like(scan)
    new_ii = (np.arange(scan.size)+integer_shift_ref) % scan.size # wraps around back to the front
    if add:
        scan_shifted[new_ii] += scan * phase_shift_ref # could vectorize but I think this would be pretty fast in Cython anyway.
    else:
        scan_shifted[new_ii] = scan * phase_shift_ref # could vectorize but I think this would be pretty fast in Cython anyway.
    return scan_shifted

def fringestop_bbdata_mpc(bbdata,delay,dest_start_time,pointing_spec,pointing_method = '1to1'):
    bbdata_out = np.zeros((bbdata.nfreq,2 * len(pointing_spec),bbdata.ntime))
    pointing_map = assign_pointing(pointing_spec,bbdata['tiedbeam_locations'],pointing_method)
    for iifreq in range(bbdata.nfreq):
        for iipointing,iibeam in enumerate(pointing_map):
            integer_shift_ref = np.rint(delay / 2.56)
            phase_shift_ref = delay - 2.56 * integer_shift_ref
            bbdata_out[iifreq,iipointing,:] = fs_ref2center_vec(
                bbdata['tiedbeam_baseband'][iifreq,iibeam,:],
                integer_shift_ref = integer_shift_ref,
                phase_shift_ref = phase_shift_ref,
                add = True)