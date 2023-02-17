"""Module which calculates scan start/end times for VLBI correlation.

Written by Calvin Leung

To define a set of scans, we must start by defining the start of a single on gate at some frequency -- call this t0 at f0 on some baseline b0.

We need to define how to extrapolate t0 as a function of baseline, frequency, and time. For each baseline, we need to have a lattice of scan start times of shape (N_freq, N_scans). Note that N_freq will vary from baseline to baseline because of different frequency coverage. The dump might also be of different length at the different stations so N_scans might be different for different baselines. However, no matter what we do, at the end of the day, we will have N_baseline lattices of shape N_freq, N_scans. The lattice defines the start of the scan as evaluated at station A. Since time does not need to be defined better than 2.56 us, we will represent all times as 64-bit floats.

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
    - Add as an option to round w_i to the next_fast_len in scipy.fft (for faster FFTs)

Finally, we need to know the "subintegration" period. After coherently dedispersing the scan, between t_ij and t_ij + w_ij, we need to know what part of the data to actually integrate. I think a reasonable way to paramterize this is with a number 0 < r_ij < 1. After coherent dedispersion, we integrate over the range t_ij + w_i/2 +- r_ij / 2 (i.e. about the central part of the scan). 
    - Warn the user if w_i * r_ij is not an integer power of 2 (for fast FFTs).
    - Add as an option to round w_i * r_ij to the next_fast_len in scipy.fft (for faster FFTs)

Now we have t_ij and w_ij for one baseline. How do we get the other baselines? They have different frequency coverage, and different time coverage. But really, all we need is to extrpolate t00 for one baseline to get t00 for another baseline, and then run the above algorithm. 
We can use difxcalc's baseline_delay function evaluated at t00 to calculate the delay between station a and station c. Then we apply the above logic to baseline cd. We ignore retarded baseline effects in this calculation but those are much smaller than the time resolution anyway.
"""

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from astropy.time import Time

from difxcalc_wrapper import io, runner, telescopes
from difxcalc_wrapper.config import DIFXCALC_CMD
from baseband_analysis.core import BBData

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)

class CorrJob():
    def __init__(self,bbdata_list):
        """Set up the correlation job:
        Given a set of BBData objects, calculate N * (N-1) / 2 baselines.
        For each baseline, define t_ij, w_ij, and r_ij into arrays of shape (N_baseline, N_freq, N_time) by calling define_scan_params().
        Use run_difxcalc and save to self.calcresults so we only call difxcalc ONCE in the whole correlator job.
        """

        self.telescopes = []
        for d in bbdata_list:
            self.telescopes.append(telescope_from_bbdata(d))# need to implement something that figures out what telescopes from the bbdata object...probably do something like BBData.index_map['input'] and figure it out.
        assert np.isclose(bbdata_list[0]['tiedbeam_locations']['ra'][:],d['tiedbeam_locations']['ra'][:]).all(),"ra values different, cannot correlate these datasets."
        assert np.isclose(bbdata_list[0]['tiedbeam_locations']['dec'][:],d['tiedbeam_locations']['dec'][:]).all(), "dec values different, cannot correlate these datasets."
        
        self.ra = bbdata_a['tiedbeam_locations']['ra'][:]
        self.dec = bbdata_a['tiedbeam_locations']['dec'][:]

        phase_centers = [
            ac.SkyCoord(ra=ra * un.deg, dec=dec * un.deg)
            for r, d in zip(R.flatten(), D.flatten())
        ]
        start_time = np.min(bbdata_a['time0']['ctime'][:])
        duration_sec = np.max(bbdata_a['time0']['ctime'][:]) - np.min(bbdata_a['time0']['ctime'][:]) + bbdata_a.ntime + bbdata_b.ntime # upper bound

        _ = io.make_calc(
            telescopes,
            phase_centers,
            start_time,
            duration_sec=int(duration_sec),
            ofile_name=calcfile,
        )
        self.calcresults = runner.run_difxcalc(
            calcfile,
            sources=phase_centers,
            telescopes=telescopes,
            force=True,
            remove_calcfile=False,
            difxcalc_cmd=DIFXCALC_CMD,
        )

        tau_at_time0 = self.calcresults.baseline_delay(
                ant1=ii_a,
                ant2=ii_b,
                time=Time(unix_a, format="unix"),
                src=src,  # should be a 1 time value x 1 pointing evaluation
            )

    def define_scan_params(t00: float, period_i, freq_offset_mode: str, time_offset_mode: str, **kwargs):
        freq = np.linspace(800,400,num = 1024, endpoint=False)
        t_ij = []
        r_ij = []
        w_ij = []

        for baseline in self.baselines:
            if freq_offset_mode == 'dm':
                _ti0 = ti0_from_t00_dm(t00,**kwargs['dm'],**kwargs['f0'])
            if freq_offset_mode == 'bbdata':
                _ti0 = _ti0_from_t00_bbdata
            _tij = tij_from_ti0_period(ti0, period_i, bbdata_a, bbdata_b)
            _rij = np.ones_like(_tij)
            _wij = wij(t_ij,r_ij,dm = dm)
        
        return t_ij, r_ij, w_ij
            
    def ti0_from_t00_dm(t00,f0,dm,fi):
        ti0 = t00 + K_DM * dm * (fi**-2 - f0**-2) # for fi = infinity, ti0 < t00.
        return ti0

    def ti0_from_t00_bbdata(bbdata_a,bbdata_b):
        unix_a = np.zeros(1024)
        unix_b = np.zeros(1024)
        unix_a[bbdata_a.index_map['freq']['id']] = bbdata_a['time0']['ctime'][:]
        unix_b[bbdata_b.index_map['freq']['id']] = bbdata_b['time0']['ctime'][:]

        ti0 = np.max(unix_a + tau_at_time0,unix_b) # use this if we fringestop A
        # ti0 = np.max(unix_a ,unix_b - tau_at_time0) # use this if we fringestop B
        # also might need to change the + to a - or vice versa
        return ti0
        
    def ti0_at_other_station(telescope):
        self.calcresults.baseline_delay(ant1 = ii_ref, ant2 = index(telescope), time = ti0, src=self.src)

    def tij_from_ti0_period(ti0,period_i,bbdata_a,bbdata_b):
        """
        ti0 : float64
            Containing start times at station A, good to 2.56 us.
        period_i : float or 1d-array
            Spacing between successive scans.
        bbdata_a : BBData object.
        """
        period_i = np.atleast_1d(period_i)
        valid_a = bbdata_a['time0']['ctime'][:] + 2.56e-6 * bbdata_a.ntime - ti0
        valid_b = bbdata_b['time0']['ctime'][:] + 2.56e-6 * bbdata_b.ntime
        n_time = np.minimum(valid_a,valid_b) / period_i
        tij = ti0[:,None] + np.arange(n_time)[None,:] * period_i[:,None]
        return tij
        
    def wij(t_ij, r_ij, dm = None): # same for every baseline
        freq = np.linspace(800,400,num = 1024, endpoint = False) # no nyquist freq
        w_ij = np.diff(t_ij,axis = -1) # the duration is equal to p_i by default
        if DM is not None: # check that wij exceeds smearing time
            smearing_time = K_DM * DM * 0.390625 / freq**3
            assert (np.max(w_ij,axis = -1) > smearing_time).all() # check all frequency channels

    def run_correlator_job(t_ij, w_ij, r_ij):
        output = VLBIVis()
        for b in baselines: # calculate cross-correlations
            for f in freq:
                answer = correlator_core(bbdata_a, bbdata_b, t_ij, w_ij, r_ij, freq_id = i) # run in series over time
                output._from_ndarray_baseline(output, freq_sel = slice(i,i+1)
        for s in stations: # calculate auto-correlations
            for f in freq:
                answer = calculate_autos(bbdata_s)
                output._from_ndarray_station(answer, freq_sel = slice(i,i+1))

        return output


    def run_correlator_job_multiprocessing(t_ij, w_ij, r_ij):
        output = VLBIVis()
        for b in baselines:
            for f in freq, but parallelized:
                answer = correlator_core(bbdata_a, bbdata_b, t_ij, w_ij, r_ij, freq_id = i) # run in series over time
                output._from_ndarray_station(answer, freq_sel = slice(i,i+1))
                # NEED SOME PARALLEL WRITER HERE
            # OR A CONCATENATE OVER FREQUENCY STAGE HERE
        
        for s in stations: # calculate auto-correlations
            for f in freq, but parallelized:
                _out_this_bl = VLBIVis()
                answer = calculate_autos(bbdata_s)
                output._from_ndarray_station(answer, freq_sel = slice(i,i+1))
                # NEED SOME PARALLEL WRITER HERE
                deep_group_copy(_out_this_bl, output)
            # OR A CONCATENATE OVER FREQUENCY STAGE HERE
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Module for doing VLBI. In a python script, you can run:
        job = CalcJob()

        # steady source: Use a duty cycle of 1 (width = 1) and integrate over the full gate (subwidth = 1)
        t_ij, w_ij, r_ij = job.define_scan_params(t00: unix_time, period_i = 0.005 sec, freq_offset_mode = 'bbdata', time_offset_mode = 'period', dm = 500)

        # pulsar or FRB: Set the period to the pulsar period, use frequency offset based on dispersion measure of the pulsar, and integrate over a small fraction of the full gate (subwidth = 0.2):
        t_ij, w_ij, r_ij = job.define_scan_params(t00: unix_time, period_i = 0.005 sec, freq_offset_mode = 'dm', time_offset_mode = 'period', dm = 500, width = 1, subwidth = 0.5)
        
        # make any modifications to t_ij, w_ij, or r_ij as necessary in your Jupyter notebook...

        # ...then call:

        run_correlator_job(t_ij, w_ij, r_ij)
        

        And go make coffee."""
    )
    parser.add_argument(
        "t_00",
        help="directory and path that holds the extracted maser data",
        type=float,
    )
    parser.add_argument(
        "time_offset",
        help="Time between scans, in seconds, measured topocentrically at the reference station. You can also specify this as a np.ndarray of 1024 numbers, but this isn't supported from the command line",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "freq_offset_mode",
        help='How to calculate frequency offset? (Options: "bbdata", "dm")'
        default="bbdata",
    )
    cmdargs = parser.parse_args()
    t_ij, w_ij, r_ij = job.define_scan_params(t00: unix_time, period_i = 0.005 sec, freq_offset_mode = 'bbdata', time_offset_mode = 'period', dm = 500)
    if parallel:
        run_correlator_job_multiprocessing(t_ij, w_ij, r_ij)
    else:
        run_correlator_job(t_ij, w_ij, r_ij)
