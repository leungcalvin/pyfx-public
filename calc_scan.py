"""Module which calculates scan start/end times for VLBI correlation.

Written by Calvin Leung

To define a set of scans, we must start by defining the start of a single on gate at some frequency -- call this t0 at f0 on some baseline b0.

We need to define how to extrapolate t0 as a function of baseline, frequency, and time. For each baseline, we need to have a lattice of scan start times of shape (N_freq, N_scans). Note that N_freq will vary from baseline to baseline because of different frequency coverage. The dump might also be of different length at the different stations so N_scans might be different for different baselines. However, no matter what we do, at the end of the day, we will have N_baseline lattices of shape N_freq, N_scans. The lattice defines the start of the scan as evaluated at station A.

Let's start by doing all the scans for one baseline, then extrapolate to the other baselines.

For one baseline, to extrapolate as a function of frequency (this is the easiest step), I can imagine two algorithms which we would be interested in.
    - "follow the DM of a pulsar/FRB" : this is used for pulsars and FRBs.
        The formula for this algorithm is simple: t_00 - t_i0 = K_DM * DM * (f0**-2 - fi**-2)

    - "start the scan ASAP" : this is used for steady sources. It makes the most use of the data dumped in each frequency channel and doesn't waste data.
        The algorithm to do this is to use the start times (BBData.index_map['time0') from each station. Call this A_i and B_i for stations A and B respectively. We don't need t_00. Instead, just use t_i0 = max(A_i, B_i + 2.56 * round(baseline_delay(A,B,t_i0,ra,dec))).

At this point, we now have t_i0 for one baseline. Now we need t_ij from t_i0 for that baseline. What are some algorithms to do this? 
    - "periodic gate start times" : this is used for slow pulsars and FRBs (where off pulses are nice). If we want to use larger gates in different frequency channels, which might result from a different pulse shape in different channels, we might want to space things out differently for different channels. We need to define the spacing between successive starts: call this P_i. Then t_ij = t_i0 + P_i * j. We should choose N_scans such that we do not exceed the dump end time for that channel. Since we need the same number of scans, we need to calculate N_scan ahead of time.

    - "pulsar binning" : this is used for millisecond pulsars and takes into account arbitrary acceleration/deceleration of the pulsar by using the topocentric rotational phase phi(i,t) at station A. Given t_i0, calculate t_ij by looking at the topocentric rotational phase phi(i,t). The rotational phase depends on the frequency (i) because of the DM delay and the time (t) because of astrophysics. Define t_i1, such that phi(i,t_ij) = phi(i,ti0) + 1 (i.e. when the pulsar rotates by one revolution.
   
Now we have t_ij, the start time as a function of time and frequency for a single baseline. Let's calculate the stop time of the scan (equivalently the width w_ij) of the scan. 
    - The only reasonable scheme I can imagine is having w_ij = w_i. 
    - Warn the user if w_i > P_i (i.e. the scans overlap).
    - Warn the user if w_i < intrachannel smearing time (K_DM * DM * df / f**3)

Finally, we need to know the "subintegration" period. After coherently dedispersing the scan, between t_ij and t_ij + w_ij, we need to know what part of the data to actually integrate. I think a reasonable way to paramterize this is with two numbers r_ij < s_ij. If r_ij = 0 and s_ij = 1, then we integrate the whole scan. If r_ij and s_ij are other numbers, we integrate up over part of the scan only (to select the pulse).
    - Warn the user if w_i * (s_ij - r_ij) is not an integer power of 2 (for fast FFTs).

Now we have t_ij and w_ij for one baseline. How do we get the others? They have different frequency coverage, and different time coverage. But really, all we need is to extrpolate t00 for one baseline to get t00 for another baseline, and then run the above algorithm. 
We can use difxcalc's baseline_delay function evaluated at t00 to calculate the delay between station a and station c. Then we apply the above logic to baseline cd. Note that the widths remain the same.


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Module for extracting delays out of maser signal. Works with data extracted by maser_extraction.py and writes delays back to the same .h5 file. Reasonable usage: python maser_delays.py /path/to/maser_data.h5"
    )
    parser.add_argument(
        "t_00",
        help="directory and path that holds the extracted maser data",
        type=str,
    )

    parser.add_argument(
        "clock",
        help='Which clock? (Options: "chime", "pathfinder", "TONE")'
    )
    parser.add_argument(
        "-d",
        "--drift",
        help="Optional: specify a drift value (in ns/day)",
        default="estimate",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Optional: allow overwriting previously-processed delays",
        action="store_true",
        default=False,
    )
    cmdargs = parser.parse_args()
    plot_dir, filename = os.path.split(cmdargs.maser_file)
    if cmdargs.drift == "estimate":
        delays_to_file(
            cmdargs.maser_file, cmdargs.clock, drift_ns_day="estimate", overwrite=cmdargs.overwrite
        )
    else:  # use actual value provided
        drift_ns_day = float(cmdargs.drift)
        delays_to_file(
            cmdargs.maser_file, cmdargs.clock, drift_ns_day=drift_ns_day, overwrite=cmdargs.overwrite
        )
