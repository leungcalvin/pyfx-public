from pyfx.math import fft_corr


def autocorr_core(DM, bbdata_A, T_A, Window, R, max_lag=None):
    """Correlates and downselects over lag (potentially more lags at shorter integration times"""
    n_freq = bbdata_A.freq
    n_time = T_A.shape
    vis_shape = (n_freq, 2, 2, 2 * max_lag + 1, n_time)
    # autocorr = np.zeros((n_freq, n_pol, n_pol, n_lag, n_time))
    auto_vis = np.zeros(vis_shape, dtype=bbdata_A.dtype)
    if max_lag is None:
        max_lag = np.max(
            window
        )  # in order to hold all autocorrelations, there must be one max lag for all frequencies and times.

    for iifreq, freq_id in enumerate(bbdata_A.index_map["freq"]["id"]):
        for iipol in range(2):
            for jjpol in range(2):
                for iitime in range(n_time):
                    t_ij = T_A[iifreq, iitime]
                    _vis = fft_corr(
                        bbdata_A[
                            iifreq,
                            :,
                            t_ij + w_ij // 2 - r_ij // 2 : t_ij + w_ij // 2 + r_ij // 2,
                        ],
                        bdata_A[
                            iifreq,
                            :,
                            t_ij + w_ij // 2 - r_ij // 2 : t_ij + w_ij // 2 + r_ij // 2,
                        ],
                        axis=-1,
                    )
                    auto_vis[iifreq, iipol, jjpol, :, iitime] = max_lag_slice(
                        _vis, max_lag, lag_axis=-1
                    )
    return auto_vis


def crosscorr_core(
    DM, bbdata_A, bbdata_B, T_A, Window, R, calc_results, index_A=0, index_B=1
):
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

    # initialize output autocorrelations and cross correlations
    shape_output_data = (
        pointing,
        bbdata_A.nfreq,
        len(time_chunks),
        2,
        2,
        time_delay_num,
    )
    autos_A = np.zeros(shape_output_data)
    autos_B = np.zeros(shape_output_data)
    cross = np.zeros(shape_output_data)
    freq = np.linspace(800, 400, num=1024, endpoint=False)

    for i in range(1024):
        f0 = freq
        for pointing in pointings:
            # Calculate tau_ij for one pointing, for all ij in that pointing.
            geodelay_ij = calcresults.retarded_baseline_delay(
                ant1=index_A, ant2=index_B, time=T_A[i, j], src=pointing
            )
            ### geodelay_0 > 0 if signal arrives at telescope A before B, otherwise geodelay_0 will be < 0.
            for j in time_chunks:

                ## get array of geometric delay over the scan (i.e .as a function of time)
                geodelay_t = calcresults.retarded_baseline_delay(
                    ant1=index_A, ant2=index_B, time=scan_times, src=pointing
                )

                ### Big fringestop (tau evaluated at t_ij)
                scan_a, scan_b_fs = get_aligned_scans(
                    bbdata_A, bbdata_B, t_ij_a, w_ij, geodelay_t[i, j], freq_id=freq[i]
                )

                ### Time dependent fringestop (tau evaluated between values of t_ij
                aligned_b *= np.exp(2j * f0 * np.pi * geodelay_t)

                #################################################
                ### It's now intrachannel de-dispersion Time. ###
                scan_a_cd = intrachannel_dedisp(scan_a, DM, f0=f0)
                scan_b_fs_cd = intrachannel_dedisp(scan_b_fs, DM, f0=f0)

                #######################################################
                ### Now that the pulses are centered at zero, calculate
                ### the start and stop time indices for on-signal ######
                start = int((w_ij - r_ij) // 2)
                stop = int((w_ij + r_ij) // 2)

                #######################################################
                ########## cross-correlate the on-signal ##############
                for pol_0 in range(2):
                    for pol_1 in range(2):
                        _vis = fft_corr(
                            scan_a_cd[pol_0, start:stop],
                            scan_b_fs_cd[pol_1, start:stop],
                            axis=-1,
                        )
                        cross[pointing, :, j, pol_0, pol_1, :] = max_lag_slice(
                            _vis, max_lag=max_lag, lag_axis=-1
                        )

        return cross


def intrachannel_dedisp(data, f0):
    """Intrachannel dedispersion: brings data to center of channel.

    This is Eq. 5.17 of Lorimer and Kramer 2004 textbook, but ONLY the last term (proportional to f^2), not the other two terms (independent of f and linearly proportional to f respectively).

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    TODO: use numba or GPU for this!"""
    n = data.shape[-1]
    f = np.fftfreq(n) * 0.390625
    for iifreq, _f0 in enumerate(f0):
        transfer_func = np.exp(
            -2j * np.pi * K_DM * DM * f**2 / _f0**2 / (f + _f0)
        )  # double check this minus sign -- might be a + instead in CHIME data.
        data[iifreq, ...] = ifft(fft(data[iifreq, ...], axis=-1) * transfer_func)
    return data


def frac_samp_shift(data, f0, tau0=None):
    """Fractional sample correction: coherently shifts data within a channel.

    data : np.ndarray of shape (nfreq,...,ntime)

    f0 : np.ndarray of shape (nfreq) holding channel centers.

    Applies a fractional phase shift of the form exp(2j*pi*(f0 + f)*tau0) to the data."""
    n = data.shape[-1]
    f = np.fftfreq(n) * 0.390625
    for iifreq, _f0 in enumerate(f0):
        transfer_func = np.exp(2j * np.pi * f * _tau0)  # apply dphi/dfreq
        data[iifreq, ...] = ifft(
            fft(data[iifreq, ...], axis=-1) * transfer_func
        ) * np.exp(
            2j * np.pi * _f0 * _tau0
        )  # apply phi
    return data


def get_aligned_scans(bbdata_a, bbdata_b, t_ij_a, w_ij, tau_at_start, freq_id):
    """For a single frequency corresponding to a given FPGA freq_id, returns aligned scans of data for that freq_id out of two provided BBData objects.

    bbdata_a : BBData
        A BBData object, with arbitrary frequency coverage.

    bbdata_b : BBData
        A BBData object, with arbitrary frequency coverage. We apply a sub-frame phase rotation with fractional sample correction to data extracted out of bbdata_b.

    t_ij_a : np.array of shape (1024, n_time)
        An array of Unix times derived with the same fractional timestamp as time0['ctime'] for bbdata_a. That is, (t_ij_a - bbdata_a['time0']['ctime']) / 2.56 = an integer.

    w_ij : int
        A particular value w_ij for this baseline. Should be an integer, and brownie points for a good FFT length.

    tau_at_start : np.float
        A delay in microseconds to apply to BBData_b, corresponding to the retarded baseline delay tau_ab evaluated at time t_ij_a.

    freq_id : int
        0 <= freq_id < 1024

    Returns
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
    iif_a = np.index(bbdata_a.index_map["freq"]["id"][:], freq_id)
    iif_b = np.index(bbdata_b.index_map["freq"]["id"][:], freq_id)

    # Use astropy to do this calculation:
    t0_a = astropy.Time(
        bbdata_a["time0"]["ctime"][iif_a], val2=bbdata_a["time0"]["ctime_offset"][iif_a]
    )
    t0_b = astropy.Time(
        bbdata_b["time0"]["ctime"][iif_b], val2=bbdata_b["time0"]["ctime_offset"][iif_b]
    )

    # assuming tau_at_start = TOA_a - TOA_b, we do
    time_we_want_at_b = t_ij_a + tau_at_start
    index_we_have_at_a = np.round((t_ij_a - t0_a) / 2.56e-6)
    index_we_have_at_b = np.round((time_we_want_at_b - t0_b) / 2.56e-6)
    time_we_have_at_b = t0_b + index_we_have_at_b * 2.56e-6
    residual = time_we_want_at_b - time_we_have_at_b
    aligned_a = bbdata_a[t0_a : t0_a + wij]
    aligned_b = frac_samp_shift(
        bbdata_b[index_we_have_at_b : index_we_have_at_b + wij],
        f0=bbdata_b.index_map["freq"]["centre"][iif_b],
        tau0=residual,
    )
    return aligned_a, aligned_b
