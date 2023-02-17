
def correlation_core(DM, bbdata_A, bbdata_B, T_A, Window, R, calc_results,index_A=0,index_B=1):
    """ 
    DM - for steady sources, set dispersion measure to 0. 
    bbdata_A - telescope A, (sliced into a frequency chunk)
    bbdata_B - telescope B, (sliced into a frequency chunk)
    T_A[i,j] - start times at ith frequency, for jth time chunk, for telescope A
    Window[i,j] - length of time chunk window (us)
    R[i,j] - fraction of time chunk (defines pulse window). Variable name should be more descriptive
    calc_results - difxcalc object containing 
    index_A - where telescope A corresponds to in calc_results 
    index_B - where telescope B corresponds to in calc_results 

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
            # make sure integer delay doesn't change as a function of time:  
            # ensure w_ij is less than the minimum amount of time it takes for integer delay to change
            c_light=300 #m/us
            rotation_rate=460/10**6 #m/us)
            assert(2*rotation_rate*Window[i,j]/c_light<2.56) #us
            #or equivalently assert w_ij<.85 sec

            geodelay_0 = calcresults.retarded_baseline_delay(ant1=index_A,  
                        ant2=index_B,
                        time=T_A[i,j],
                        src=pointing)
            ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0. 

            ### revisit revisit (rounding or floor) #####
            sign=np.sign(geodelay_0)
            int_geodelay_0 = sign*np.floor((sign*geodelay_0).to(un.us).value / 2.56, dtype=int) ## floor absolute value. This might be stupid. 

            tstart_b=T_A[i,j]+int_geodelay_0
            assert(tstart_b in bbdata_B['times']) #revisit this case later

            ## assuming some def time_slice(data,start time, end time) 
            bbdata_A_clipped=time_slice(bbdata_A,T_A[i,j],T_A[i,j]+Window[i,j])
            
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

            #################################################
            ### It's now intrachannel de-dispersion Time. ###
            bbdata_A_dedispersed=coherent_dedisp(bbdata_A, DM)
            bbdata_B_dedispersed=coherent_dedisp(bbdata_B, DM)

            #######################################################
            ### Now that the pulses are centered at zero, calculate
            ### the start and stop time indices for on signal ######

            on_signal_start=center_scan_time-R[i,j]*int(Window[i,j]/ 2.56)/2
            on_signal_stop=center_scan_time+R[i,j]*int(Window[i,j]/ 2.56)/2


            bbdata_A_signal_on=bbdata_A_dedispersed[:,:,on_signal_start:on_signal_stop]
            bbdata_B_signal_on=bbdata_B_dedispersed[:,:,on_signal_start:on_signal_stop]
            
            
            for pol_1 in pols:  
                autos_A[pointing,:, j, pol_1, pol_1, :] = ifft(fft(bbdata_A_signal_on[pol_1],bbdata_A_signal_on[pol_1],axis=-1)
                autos_B[pointing,:, j, pol_1, pol_1, :] = ifft(fft(bbdata_B_signal_on[pol_1],bbdata_B_signal_on[pol_1],axis=-1)
                for pol_2 in pols:
                    autos_cross[pointing,:, j, pol_1, pol_2, :] = ifft(fft(bbdata_A_signal_on[pol_1],bbdata_B_signal_on[pol_2],axis=-1)
                    

        return cross, autos_A, autos_B

