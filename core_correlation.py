from pyfx.core_math import fft_corr
from pyfx.core_math import max_lag_slice
import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)

def autocorr_core(DM, bbdata_A, T_A, Window, R, max_lag=None,n_pol=2):
    """Autocorrelates and downselects over lag.

    Parameters
    ----------
    DM : float
        The DM with which we de-smear the data before the final gating. For steady sources, set DM to 0. Only the zeroth pointing gets de-dispersed!
    bbdata_A : BBData
        A BBData object to auto-correlate.
    T_A : np.ndarray
        An array of start times accurate to 2.56us, for the station being correlated.
        Should have shape (nfreq, 2*nbeam, ntime), where nbeam is the number of beams formed at that station.
    Window : np.ndarray
        An array of ints indicating correlation integration periods.
        Should have shape (nfreq, ntime).
    R : np.ndarray
        The fraction of the window that gets accumulated, centered on the data frame at T_A + Window / 2 +- R / 2.
        Should have shape (n_freq, n_po, n_time)
    max_lag : The maximum lag out to which we should correlate.
    n_pol : int
        Number of polarizations in data. Always 2.
    """
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(T_A,axis=-1)
    n_beams =bbdata_A["tiedbeam_baseband"].shape[1] // 2
    
    R = np.atleast_2d(R)
    assert R.shape[0] == 1024
    
    if max_lag is None:
        max_lag = np.max(
            Window
        )  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.
    
    vis_shape = (n_freq, n_beams, n_pol, n_pol, 2 * max_lag + 1, n_scan)
    auto_vis = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)

    ## will compress this horrific number of for loops later
    for iifreq in range(n_freq):  
        for iibeam in range(n_beams):
            for iitime in range(n_scan):
                t_ij = T_A[iifreq, iibeam, iitime] # here, the beam axis indexes the beamformer pointings.
                w_ij=Window[iifreq,iibeam, iitime] # here, the beam axis indexes the beamformer pointings.
                r_ij=R[iifreq,iibeam, iitime] # Use r_ij
                start = int((w_ij - w_ij*r_ij) // 2)
                stop = int((w_ij + w_ij*r_ij) // 2)
                for iipol in range(n_pol):
                    for jjpol in range(n_pol):
                        _vis = fft_corr(
                            bbdata_A['tiedbeam_baseband'][
                                iifreq,
                                iibeam + iipol,
                                start : stop,
                            ],
                            bbdata_A['tiedbeam_baseband'][
                                iifreq,
                                iibeam + jjpol,
                                start : stop,
                            ])
                        auto_vis[iifreq, iibeam, iipol, jjpol, :, iitime] = np.concatenate((_vis[:max_lag+1],_vis[-max_lag:]))


    return auto_vis


def crosscorr_core(bbdata_A, bbdata_B, T_A, Window, R, calc_results,DM,index_A=0, index_B=1,sample_rate=2.56,max_lag=20,n_pol=2,complex_conjugate_convention = -1, intra_channel_sign = 1):
    """Autocorrelates and downselects over lag.

    Parameters
    ----------
    bbdata_A : BBData
        A BBData object to auto-correlate (sliced into a frequency chunk).
    bbdata_B : BBData
        A BBData object to auto-correlate (sliced into a frequency chunk).
    T_A : np.ndarray
        An array of start times accurate to 2.56us, for the station being correlated.
        Should have shape (nfreq, 2*nbeam, ntime), where nbeam is the number of beams formed at that station.
    Window : np.ndarray
        An array of ints indicating correlation integration periods.
        Should have shape (nfreq, npointing, ntime).
    R : np.ndarray
        The fraction of the window that gets accumulated, centered on the data frame at T_A + Window / 2 +- R / 2.
        Should have shape (nfreq, npointing, ntime).
    max_lag : int
        The maximum lag out to which we should correlate.
    DM : float
        The DM with which we de-smear the data before the final gating. For steady sources, set DM to 0. Only the zeroth pointing gets de-dispersed!
    calc_results - difxcalc object containing
    index_A : int
        Index of telescope A corresponds in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    index_B : int
        Index of telescope B corresponds in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    sample_rate - rate at which data is sampled in microseconds. Always 2.56.

    Outputs:
    cross - array of autocorrelations and cross correlations with shape (freq, pointing, pol, pol, delay, timechunk)

    """
    n_freq = len(bbdata_A.freq)
    n_scan = np.size(T_A,axis=-1)
    n_pointings=bbdata_A["tiedbeam_baseband"].shape[1] // 2 ## SA: basing this off of how the data is arranged now, may want to change
    
    # initialize output autocorrelations and cross correlations
    
    vis_shape = (n_freq, n_pointings, n_pol, n_pol,  2 * max_lag + 1,n_scan)
    cross = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)
    for iifreq in range(n_freq):  
        f0 = bbdata_B.index_map["freq"]["centre"][iifreq] 
        f0_a = bbdata_B.index_map["freq"]["centre"][iifreq] 
        print(iifreq)
        assert f0 == f0_a, "Mismatched frequency axes!"
        T_A_index=np.rint((T_A - bbdata_A['time0']['ctime'][:,None,None]) / 2.56e-6).astype(int)
        for iipointing in range(n_pointings):
            for jjscan in range(n_scan):
                w_ij=Window[iifreq,iipointing,jjscan]
                r_ij=R[iifreq,iipointing,jjscan]
                t0_a = bbdata_A["time0"]["ctime"][iifreq]+T_A_index[iifreq,iipointing,jjscan]*(sample_rate*1e-6)            
                ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0.
                ## get array of geometric delay over the scan (i.e .as a function of time)
                
                start_time= Time(
                    t0_a,
                    val2=bbdata_A["time0"]["ctime_offset"][iifreq],
                    format="unix",
                    precision=9,  
                )
                query_times = start_time + sample_rate*1e-6 * un.s * (T_A_index[iifreq,iipointing,jjscan]+np.arange(w_ij)) #also calculating 100 frames ahead as cushion
                # using telescope A times as reference time
                ### should probably just call this once out of the for loop using np.flatten but will fix later
                geodelay = calc_results.retarded_baseline_delay(
                    ant1=index_A, ant2=index_B, time=query_times, src=iipointing,delay_sign=0,self_consistent=False
                )           
                
                ### Fringestopping B -> A  
                scan_a, scan_b_fs = get_aligned_scans(
                    bbdata_A, bbdata_B,T_A_index[iifreq,iipointing,jjscan], w_ij, geodelay, freq_id=iifreq,intra_channel_sign = intra_channel_sign, sample_rate=sample_rate
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
                        cross[iifreq, iipointing, pol_0, pol_1, :max_lag+1, jjscan] = _vis[:max_lag+1]
                        cross[iifreq, iipointing, pol_0, pol_1, -max_lag:, jjscan] = _vis[-max_lag:]
        return cross


def intrachannel_dedisp(data, DM,f0,sample_rate=2.56):
    """Intrachannel dedispersion: brings data to center of channel.

    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook, but ONLY the last term (proportional to f^2), not the other two terms (independent of f and linearly proportional to f respectively).

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    sample_rate : sampling rate of data in microseconds

    TODO: use numba or GPU for this!"""
    if DM == 0:
        return data
    else:
        n = data.shape[-1]
        f = np.fft.fftfreq(n) * sample_rate
        transfer_func = np.exp(-2j * np.pi * K_DM * DM * f**2 / f0**2 / (f + f0))  # double check this minus sign -- might be a + instead in CHIME data.
        data_cd = np.fft.ifft(np.fft.fft(data, axis=-1) * transfer_func)
        return data_cd


def frac_samp_shift(data, f0, sub_frame_tau=None,sample_rate=2.56,freq_id=None, complex_conjugate_convention=-1,
intra_channel_sign=1):
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
    transfer_func = np.exp(intra_channel_sign*2j * np.pi * f * np.median(sub_frame_tau))  # apply dphi/dfreq
    data2 = np.fft.ifft(
        np.fft.fft(data) * transfer_func
    ) * np.exp(
        complex_conjugate_convention*2j * np.pi * f0 * sub_frame_tau
    )  # apply phi
    return data2
    
                
def get_aligned_scans(bbdata_A, bbdata_B,T_A_index, wij, tau,freq_id,intra_channel_sign, sample_rate=2.56):
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.

    bbdata_A : BBData
        A BBData object, with arbitrary frequency coverage.

    bbdata_B : BBData
        A BBData object, with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    T_A_index : np.array of shape (1024, n_time)
        An array of indices corresponding to the start frames for telescope A 

    w_ij : int
        A particular value w_ij for this baseline. Should be an integer, and brownie points for a good FFT length.

    int_delay0 : np.float
        A delay in frames to apply to BBData_b, corresponding to the retarded baseline delay tau_ab evaluated at time t_ij_a rounded to the nearest integer frame number.

    tau : np.array (n_frame) of dtype np.float
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
        Number of frames by which we need to shift T_A_ij in order to ensure T_A_ij+geodelay is contained within bbdata_B. Note that in the event that geodelay is positive, newstart will always be 0 (assuming the user has chosen T_A_ij such that the unix time is in both datasets)

    Super technical note on floor vs round: it doesn't matter AS LONG AS you do a sub-sample rotation (or better yet, a frac samp correction)! Suppose your total delay is 10.6 frames.
    - You can round to 11 frames. You should keep track that you rounded to 11, and then do frac samp -0.4.
    - You can floor to 10 frames, you should keep track that you floored to 10, and then do frac samp +0.6.

    Answer should be the same either way -- as long as you do the frac samp correction!
    After doing the integer part (shift by either 10 or 11 frames), we need to apply a phase rotation. Note that exp(2j*np.pi*channel_center * -0.4/(2.56us) = exp(2j*np.pi*channel_center * +0.6/(2.56us), for the exact frequency corresponding to channel center, but not for any of the other frequencies that do not satisfy f = 800 - (N * 0.390625 MHz) for integers N -- this is the narrowband approximation. We experience some de-correlation near the band edges, which is why we use fractional sample correction in this code.
    """

    time_we_want_at_b = tau[0] #us
    aligned_a = np.zeros((2,wij),dtype = complex)
    aligned_a[:,0:min(T_A_index + wij, bbdata_A.ntime) - T_A_index] = bbdata_A['tiedbeam_baseband'][freq_id,...,T_A_index:T_A_index + wij]
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

    # the next four lines are indices into the BBData.
    src_start_want = T_A_index+int_delay #frame number closest to start time
    src_stop_want = src_start_want + wij

    # if no data, throw out
    if src_stop_want < 0 or src_start_want > bbdata_B.ntime:
        print('WARNING: Scan requested with no valid data present')
        return aligned_a, np.zeros_like(aligned_a)

    src_start_have = np.max([src_start_want, 0])
    src_stop_have = np.min([src_stop_want,bbdata_B.ntime])

    w_ij_have =int(src_stop_have-src_start_have)
    correction_factor = wij / w_ij_have # if you are missing half the data, multiply by 2.
    if correction_factor > 2:
        # warn the user that the boundary conditions are sketch if we are missing e.g. more than half the data.
        print(f"WARNING: over half the data is missing for {bbdata_B.index_map['input'][0]}")
        print(src_start_want)
        print(src_stop_want)
        print(src_start_have)
        print(src_stop_have)
    dest_start_idx = src_start_have - src_start_want # this is >= 0...
    copy_duration = src_stop_have - src_start_have # also >= 0
    dest_stop_idx = dest_start_idx + copy_duration
    aligned_b = np.zeros_like(aligned_a)
    aligned_b[...,dest_start_idx:dest_stop_idx] = bbdata_B['tiedbeam_baseband'][freq_id,...,src_start_have:src_stop_have] * correction_factor 
    
    time_we_have_at_b = (delta_A_B+int_delay*sample_rate*1e-6) #s
    sub_frame_tau=(tau[0] - time_we_have_at_b*1e6) #sub-frame delay at start time in mircoseconds
    
    aligned_b = frac_samp_shift(aligned_b,
        f0=bbdata_B.index_map["freq"]["centre"][freq_id],
        sub_frame_tau=sub_frame_tau,
        sample_rate=sample_rate,
        freq_id=freq_id,
        intra_channel_sign = intra_channel_sign)
    return aligned_a, aligned_b
