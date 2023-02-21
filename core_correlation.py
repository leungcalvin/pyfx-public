from pyfx.math import fft_corr

def autocorr_core(DM, bbdata_A, T_A, Window, R, max_lag = None):
    """Correlates and downselects over lag (potentially more lags at shorter integration times"""
    n_freq = bbdata_A.freq
    n_time = T_A.shape
    vis_shape = (n_freq, 2, 2, 2 * max_lag + 1, n_time)
    #autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))
    auto_vis = np.zeros(vis_shape,dtype = bbdata_A.dtype)
    if max_lag is None: 
        max_lag = np.max(window) # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.
    
    for iifreq, freq_id in enumerate(bbdata_A.index_map['freq']['id']):
        for iipol in range(2):
            for jjpol in range(2):
                for iitime in range(n_time):
                    t_ij = T_A[iifreq, iitime]
                    _vis = fft_corr(bbdata_A[iifreq, :, t_ij + w_ij // 2 - r_ij // 2: t_ij + w_ij // 2 + r_ij // 2], 
                                    bdata_A[iifreq, :, t_ij + w_ij // 2 - r_ij // 2: t_ij + w_ij // 2 + r_ij // 2], 
                                    axis = -1)
                    auto_vis[iifreq, iipol, jjpol, :, iitime] = max_lag_slice(_vis, max_lag, lag_axis = -1)
    return auto_vis
        
def crosscorr_core(DM, bbdata_A, bbdata_B, T_A, Window, R, calc_results,index_A=0,index_B=1):
    """ 
    DM - the DM with which we de-smear the data before the final gating. for steady sources, set dispersion measure to 0. 
    bbdata_A - telescope A, (sliced into a frequency chunk)
    bbdata_B - telescope B, (sliced into a frequency chunk)
    T_A[i,j] - start times at ith frequency, for jth time chunk, for telescope A
    Window[i,j] - length of time chunk window (us)
    R[i,j] - fraction of time chunk (defines pulse window). Variable name should be more descriptive
    calc_results - difxcalc object containing 
    index_A - where telescope A corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object.
    index_B - where telescope B corresponds to in calc_results: CL: ideally, you should figure out the telescope index from the BBData object. 

    Note: t_ij_B is not used here
    
    "scan" = "time chunk" = "window"
    for steady sources, "on window" = "window" (R=1)

    outputs array of autocorrelations and cross correlations with shape (freq, timechunk, pol, pol, delay)

    ### check lorentz math for DM 
    """
    
    ## need to decide how many time delays we want to return 

    #initialize output autocorrelations and cross correlations
    shape_output_data=(pointing,bbdata_A.nfreq,len(time_chunks),2, 2,time_delay_num)
    autos_A = np.zeros(shape_output_data)
    autos_B = np.zeros(shape_output_data)
    cross = np.zeros(shape_output_data)

    for pointing in pointings:
        for j in time_chunks: 

            #################################################
            ### Calculate scanning window for A and B #######
            geodelay_0 = calcresults.retarded_baseline_delay(ant1=index_A,  
                        ant2=index_B,
                        time=T_A[i,j],
                        src=pointing)
            ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0. 

            ### revisit revisit (rounding or floor) #####
            ### CL: having thought about this more, I don't think it matters, AS LONG AS you do the frac samp correction. 
            # Suppose your total delay is 10.6 frames.
            # You can round to 11 frames. You should keep track that you rounded to 11, and then do frac samp -0.4.
            # If you floor to 10 frames, you should keep track that you floored to 10, and then do frac samp +0.6.

            # Answer should be the same either way -- as long as you do the frac samp correction!

            # Similarly, if you apply phase, exp(2j*np.pi*channel_center * -0.4/(2.56us) = exp(2j*np.pi*channel_center * +0.6/(2.56us), but only for the exact frequency corresponding to channel center, not any of the other frequencies that do not satisfy f = 800 - (N * 0.390625 MHz) for integers N -- this is the narrowband approximation.
            
            sign=np.sign(geodelay_0)
            int_geodelay_0 = sign*np.floor((sign*geodelay_0).to(un.us).value / 2.56, dtype=int) ## floor absolute value. This might be stupid. 

            tstart_b=T_A[i,j]+int_geodelay_0
            assert(tstart_b in bbdata_B['times']) #revisit this case later

            ## assuming some def time_slice(data,start time, end time) 
            bbdata_A_clipped=time_slice(bbdata_A,T_A[i,j],T_A[i,j]+Window[i,j])
            
            
            #################################################
            ############ Fringestop bbdata_B  ###############
            
            ## fringestop bbdata B to nearest integer frame (the scan window should be short enough so that int_geodelay_0 is constant within the time chunk) 
            bbdata_B_clipped=time_slice(tstart_b,tstart_b+Window[i,j])

            # I *think* for now we want these in terms of A times 
            scan_times = bbdata_A_clipped['times'] # or something like times=T_A[i,j] + 2.56e-6 * un.s * np.range(int(Window[i,j]/(2.56e-6)))

            ## get array of geometric delay over the scan (i.e .as a function of time)
            geodelay_t = calcresults.retarded_baseline_delay(ant1=index_A,  
                ant2=index_B,
                time=scan_times,
                src=pointing)

            ### fringestop bbdata B using subinteger delay
            subint_geo_t=geodelay_t-int_geodelay_0

            subint_delay_phase=2* np.pi* subint_geo_t* bbdata_B_clipped.index_map["freq"]["centre"]
            bbdata_B_clipped=bbdata_B_clipped* np.exp(1j * subint_delay_phase)
            ## will add fractional sample correction later
            ## Could consider applying fractional sample correction NOW via get_aligned_scans(), then apply subint_delay_phase and keep going.

            
            #################################################
            ### It's now intrachannel de-dispersion Time. ###
            bbdata_A_dedispersed=intrachannel_dedisp(bbdata_A, DM)
            bbdata_B_dedispersed=intrachannel_dedisp(bbdata_B, DM)

            
            #######################################################
            ### Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on-signal ######

            on_signal_start=center_scan_time-R[i,j]*int(Window[i,j]/ 2.56)/2
            on_signal_stop=center_scan_time+R[i,j]*int(Window[i,j]/ 2.56)/2

            bbdata_A_signal_on=bbdata_A_dedispersed[:,:,on_signal_start:on_signal_stop]
            bbdata_B_signal_on=bbdata_B_dedispersed[:,:,on_signal_start:on_signal_stop]
            
            #######################################################
            ########## cross-correlate the on-signal ##############
            for pol_1 in pols:  
                autos_A[pointing,:, j, pol_1, pol_1, :] = ifft(fft(bbdata_A_signal_on[pol_1],bbdata_A_signal_on[pol_1],axis=-1)
                autos_B[pointing,:, j, pol_1, pol_1, :] = ifft(fft(bbdata_B_signal_on[pol_1],bbdata_B_signal_on[pol_1],axis=-1)
                for pol_2 in pols:
                    autos_cross[pointing,:, j, pol_1, pol_2, :] = ifft(fft(bbdata_A_signal_on[pol_1],bbdata_B_signal_on[pol_2],axis=-1)
                    

        return cross, autos_A, autos_B

def intrachannel_dedisp(data, f0, tau0 = None):
    """Intrachannel dedispersion: brings data to center of channel.

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    TODO: use numba or GPU for this!"""
    n = data.shape[-1]
    f = np.fftfreq(n) * 0.390625
    for iifreq, _f0 in enumerate(f0):
        transfer_func = np.exp(-2j*np.pi*K_DM * DM * f**2 / _f0**2 / (f + _f0)) # double check this minus sign -- might be a + instead in CHIME data.
        data[iifreq,...] = ifft(fft(data[iifreq,...],axis = -1) * transfer_func)
    return data

def frac_samp_shift(data, f0, tau0 = None):
    """Fractional sample correction: coherently shifts data within a channel.

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    TODO: use numba or GPU for this!"""
    n = data.shape[-1]
    f = np.fftfreq(n) * 0.390625
    for iifreq, _f0 in enumerate(f0):
        transfer_func = np.exp(2j*np.pi*f*_tau0) # apply dphi/dfreq
        data[iifreq,...] = ifft(fft(data[iifreq,...],axis = -1) * transfer_func) * np.exp(2j*np.pi*_f0*_tau0) # apply phi
    return data
        
def get_aligned_scans(bbdata_a,bbdata_b, t0_a,t0_b,tau_at_start):
    # One freq only, since aligned_a and aligned_b are probably diferent lengths for diff frequencies. Could zero pad aligned_a and aligned_b.
    # Use astropy to do this calculation:
    t0_a = astropy.Time(bbdata_a['time0']['ctime'][:],val2 = bbdata_a['time0']['ctime_offset'][:])
    t0_b = astropy.Time(bbdata_b['time0']['ctime'][:],val2 = bbdata_b['time0']['ctime_offset'][:])

    # assuming tau_at_start = TOA_a - TOA_b, we do
    time_we_want_at_b = t0_a + tau_at_start
    index_we_have_at_b = np.round((time_we_want_at_b - t0_b ) / 2.56e-6)
    time_we_have_at_b = t0_b + index_we_have_at_b * 2.56e-6
    residual = (time_we_want_at_b - time_we_have_at_b)
    aligned_a = bbdata_a[t0_a:t0_a + wij]
    aligned_b = frac_samp_shift(bbdata_b[index_we_have_at_b:index_we_have_at_b + wij],f0 = bbdata_a.index_map['freq']['centre'][:],tau0 = residual)

    return aligned_a,aligned_b
