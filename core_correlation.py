from pyfx.core_math import fft_corr
from pyfx.core_math import max_lag_slice
import numpy as np
from astropy.time import Time, TimeDelta
from decimal import Decimal
import astropy.units as un
import scipy 
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
    n_pointings=bbdata_A["tiedbeam_baseband"].shape[1] // 2 ## SA: basing this off of how the data is arranged now, may want to change
    n_freq = bbdata_A.freq
    n_scan = np.size(T_A,axis=-1)
    if max_lag is None:
        max_lag = np.max(
            window
        )  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.
    
    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))
    auto_vis = np.zeros(vis_shape, dtype=bbdata_A.dtype)

    for iifreq, freq_id in enumerate(bbdata_A.index_map["freq"]["id"]):
        for pointing in n_pointings:
            for iipol in range(n_pol):
                for jjpol in range(n_pol):
                    for iitime in range(n_scan):
                        t_ij = T_A[iifreq, iitime]
                        _vis = fft_corr(
                            bbdata_A['tiedbeam_baseband'][
                                iifreq,
                                iipol,
                                t_ij + w_ij // 2 - r_ij // 2 : t_ij + w_ij // 2 + r_ij // 2,
                            ],
                            bdata_A['tiedbeam_baseband'][
                                iifreq,
                                jjpol,
                                t_ij + w_ij // 2 - r_ij // 2 : t_ij + w_ij // 2 + r_ij // 2,
                            ],
                            axis=-1,
                        )
                        auto_vis[iifreq, pointing, iipol, jjpol, iitime,:] = max_lag_slice(
                            _vis, max_lag, lag_axis=-1
                        )
    return auto_vis


def crosscorr_core(bbdata_A, bbdata_B, T_A, Window, R, calc_results,DM,index_A=0, index_B=1,sample_rate=2.56,max_lag=None,n_pol=2):
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
    n_scan = np.size(T_A,axis=-1)
    n_pointings=bbdata_A["tiedbeam_baseband"].shape[1] // 2 ## SA: basing this off of how the data is arranged now, may want to change
    
    # initialize output autocorrelations and cross correlations
    if max_lag is None:
        max_lag = 100  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.
    
    vis_shape = (n_freq, n_pointings, n_pol, n_pol, n_scan, 2 * max_lag + 1)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))

    cross = np.zeros(vis_shape, dtype=bbdata_A['tiedbeam_baseband'].dtype)

    if (type(R) is np.ndarray)==False:
        R=np.full(Window.shape, R)

    for pointing in range(n_pointings):
        for iifreq in range(n_freq):  
            ### require user to have "well-ordered" bbdata in frequency (iifreqA=iifreqB)      
            f0 = bbdata_A.index_map["freq"]["centre"][iifreq] ##frequency centers in MHz
            for jjscan in range(n_scan):
                # Use astropy to do this calculation:
                w_ij=Window[iifreq,jjscan]
                r_ij=R[iifreq,jjscan]
                t0_a = bbdata_A["time0"]["ctime"][iifreq]
                T_A_index = int((T_A[iifreq,jjscan] - t0_a)/ (sample_rate*1e-6)) #frame number of start time, should have no remainder
            
                ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0.
                ## get array of geometric delay over the scan (i.e .as a function of time)
                
                
                '''SEA: -(in the case of steady sources) its hard to guarantee t_a exists in both datasets
                if you want to maximize the scan over the dump, need to calculate geodelay
                -use geocenter later? '''
                
                start_time= Time(
                    T_A[iifreq,jjscan],
                    val2=bbdata_A["time0"]["ctime_offset"][iifreq],
                    format="unix",
                    precision=9,  
                )
                query_times = start_time + sample_rate*1e-6 * un.s * (T_A_index+np.arange(w_ij))
                # using telescope A times as reference time
        ### should probably just call this once out of the for loop but will fix later
                geodelay = calc_results.retarded_baseline_delay(
                    ant1=index_A, ant2=index_B, time=query_times, src=pointing,self_consistent=False
                )
                geodelay0=geodelay[0]
                
                int_geodelay=int(np.round(geodelay[0]/sample_rate)) #assuming scan is short enough so that int_geodelay=int_geodelay_0
                subint_geodelay=geodelay-int_geodelay
                

                ### Fringestopping B -> A at time T_A
                scan_a, scan_b_fs,subint_geodelay0,newstart = get_aligned_scans(
                    bbdata_A, bbdata_B, T_A[iifreq,jjscan],T_A_index, w_ij, geodelay0, freq_id=iifreq,sample_rate=sample_rate
                )
                w_ij=np.size(scan_a,axis=-1) ## update width of the scan after geometric delay is applied (i.e. if original width went beyond bounds of the data, see get_aligned_scans)
                subint_geodelay=subint_geodelay[newstart:newstart+w_ij]
                ### applying time-dependent correction to geometric delay  
                scan_b_fs *= np.exp(2j * f0 * np.pi * (subint_geodelay-subint_geodelay0)) 
                

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
        return cross


def intrachannel_dedisp(data, DM,f0,sample_rate=2.56):
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
    data= np.fft.ifft(np.fft.fft(data, axis=-1) * transfer_func)
    return data


def frac_samp_shift(data, f0, tau0=None,sample_rate=2.56):
    """Fractional sample correction: coherently shifts data within a channel.

    data : np.ndarray of shape (ntime)  

    f0 : frequency channel center.

    sample_rate : sampling rate of data in microseconds

    tau0 : sub-frame delay in us 

    Applies a fractional phase shift of the form exp(2j*pi*f*tau0) to the data.
    
    ## need to rethink looping over frequency in the main function; this should take in an array of freqs
    """
    n = data.shape[-1]
    f = scipy.fft.fftfreq(n, sample_rate)
    transfer_func = np.exp(-2j * np.pi * f * tau0)  # apply dphi/dfreq
    ### This is probably NOT negative for most other telescopes??

    data = np.fft.ifft(
        np.fft.fft(data, axis=-1) * transfer_func
    ) * np.exp(
        2j * np.pi * f0 * tau0
    )  # apply phi
    return data


def get_aligned_scans(bbdata_A, bbdata_B, T_A_ij,T_A_index, wij, tau_at_start,freq_id, sample_rate=2.56):
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

    tau_at_start : np.float
        A delay in microseconds to apply to BBData_b, corresponding to the geometric delay evaluated at time t_ij_a 

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
   
    t0_b = bbdata_B["time0"]["ctime"][freq_id]

    ## We need to enure that with the integer geometric frame delay applied, the width of the scan is still within the bounds of the data
    ## first case: if the geodelay is negative (signal arrived at telescope B first), ensure that start of scan is within bounds of telescope B. 
    ## based on how we've parameterized the code, it's much easier to adjust Wij than Rij....with pulsars though it makes more sense to adjust Rij,
    ## but we would need to use np.roll or cushion the data. Revisit
    newstart=0
    time_we_want_at_b = T_A_ij + tau_at_start*1e-6 #seconds
    index_we_have_at_b = int(np.round((time_we_want_at_b - t0_b) / (sample_rate*1e-6))) #frame number closest to start time
    residual=(time_we_want_at_b - t0_b)*1e6 -index_we_have_at_b*(sample_rate) #mircoseconds
    if index_we_have_at_b<0:
        newstart=-index_we_have_at_b #the number of frames we have to shift T_A_ij by to ensure T_A_ij+geodelay is contained within bbdata_B
        wij-=newstart
        T_A_index+=newstart
        index_we_have_at_b=0

    ## we also need to ensure that with the integer geometric frame delay applied, the end of the scan is still within the bounds of the data
    max_wij_B=np.ma.size(bbdata_B["tiedbeam_baseband"],axis=-1)-index_we_have_at_b 
    max_wij_A=np.ma.size(bbdata_A["tiedbeam_baseband"],axis=-1)-T_A_index
    wij=np.min([max_wij_B,max_wij_A,wij])
    aligned_a = bbdata_A['tiedbeam_baseband'][freq_id,:,T_A_index:T_A_index + wij]
    aligned_b = frac_samp_shift(bbdata_B['tiedbeam_baseband'][freq_id,:,index_we_have_at_b :index_we_have_at_b + wij],
        f0=bbdata_B.index_map["freq"]["centre"][freq_id],
        tau0=residual,
        sample_rate=sample_rate
    )
    return aligned_a, aligned_b,residual,newstart





