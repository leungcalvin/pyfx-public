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
import os,datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.time import Time
import astropy.coordinates as ac
import astropy.units as un
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import fill_waterfall,_scrunch
from difxcalc_wrapper import io, runner, telescopes
from difxcalc_wrapper.config import DIFXCALC_CMD
from scipy.fft import fft, ifft, next_fast_len
from scipy.stats import median_abs_deviation

from decimal import Decimal
from astropy.time import Time,TimeDelta

from pyfx.core_correlation import autocorr_core, crosscorr_core
from pyfx.bbdata_io import station_from_bbdata, get_all_time0, get_all_im_freq
from pyfx.config import CALCFILE_DIR

from coda.core import VLBIVis

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
FREQ = np.linspace(800,400,num = 1024, endpoint = False)

def same_pointing(bbdata_a,bbdata_b):
    assert np.isclose(bbdata_a['tiedbeam_locations']['ra'][:],bbdata_b['tiedbeam_locations']['ra'][:]).all()
    assert np.isclose(bbdata_a['tiedbeam_locations']['dec'][:],bbdata_b['tiedbeam_locations']['dec'][:]).all()
    return True

def _ti0_even_spacing(bbdata_filename, ti0,  period_frames, num_scans_before = 0, num_scans_after = 'max',time_ordered = False):
    """
    ti0 : np.array of float64s
        Containing start times at station A, good to 2.56 us.
    period_frames : int 
        Spacing between successive scans in frames. Could be an int or an array of ints -- one per frequency.
    bbdata : BBData object.
    """
    time0 = get_all_time0(bbdata_filename)
    im_freq = get_all_im_freq(bbdata_filename)
    if num_scans_before == 'max': # attempt to use the whole dump
        start_time = extrapolate_to_full_band(time0,im_freq['id'])['ctime']
        num_scans_per_freq = (ti0 - start_time) // (2.56e-6 * period_frames)
        num_scans_before = np.max(num_scans_per_freq)

    if num_scans_after == 'max': # attempt to use the whole dump
        end_time = start_time + 2.56e-6 * bbdata.ntime
        num_scans_per_freq = (end_time - ti0) // (2.56e-6 * period_frames)
        num_scans_after = np.max(num_scans_per_freq) - 1 # minus one, since "num_scans_after" does not include the scan starting at ti0.
    scan_numbers = np.hstack((np.arange(0,num_scans_after),
                              np.arange(-num_scans_before,0))
                            ) 
    # this makes the scan number corresponding to ti0 come first in the array: e.g. if the on-pulse comes in the 5th scan period, the t_ij array is ordered as (5,6,7,...0,1,2,3,4)
    if time_ordered: # this makes the scans time-ordered, e.g. 0,1,2,3,...8,9.
        scan_numbers = sorted(scan_numbers) 

    tij = ti0[:, None] + scan_numbers[None, :] * 2.56e-6 * period_frames[:, None]
    return round_to_integer_frame(tij, bbdata_filename)

def _ti0_from_t00_bbdata(bbdata_filename, t_00,f0):
    """Returns ti0 such that it aligns with the start frame of each channel in the bbdata (potentially with a constant frame_offset > 0 for all channels).

    This always returns 1024 start times, since different baselines might have different amounts of frequency coverage.

    Parameters
    ----------
    bbdata_a : A BBData object used to define the scan start

    frame_offset : int
        An integer number of frames (must be non0-negative) to skip.

    Returns
    -------
    ti0 : np.array of shape (1024,)
    """
    im_freq= get_all_im_freq(bbdata_filename)
    t0 = get_all_time0(bbdata_filename)
    iifreq = np.argmin(np.abs(im_freq["centre"][:] - f0))
    sparse_ti0 = t0.copy()
    sparse_ti0['ctime'][:] = t0['ctime'][:] + (t_00 - t0['ctime'][iifreq]) # ti0 if these were all the frequencies we cared about. easy!

    #...but, we always need all 1024 frequencies! need to interpolate reasonably.
    ti0 = extrapolate_to_full_band(sparse_ti0, im_freq['id'][:])['ctime']
    return ti0

def _ti0_from_t00_dm(bbdata_filename, t00, f0, dm, fi):
    ti0 = t00 + K_DM * dm * (fi**-2 - f0**-2)  # for fi = infinity, ti0 < t00.
    return round_to_integer_frame(ti0, bbdata_filename)

def validate_wij(w_ij, t_ij, r_ij, dm=None):
    """Performs some following checks on w_ij
    
    Parameters
    ----------
    w_ij : np.ndarray of ints
        Integration length, in frames, as a function of (n_freq, n_pointing, n_time)

    t_ij : np.ndarray of float64
        UNIX time (in seconds) at which integration starts @ each station (n_station, n_freq, n_pointing, n_time)

    r_ij : np.ndarray of float64
        A number between 0 and 1 denoting the sub-integration, (n_freq, n_pointing, n_time).

    Returns
    -------
    True : If all the following checks pass...
        1) No overlapping sub-integrations (integrations might overlap, if the smearing timescale within a channel exceeds the pulse period
        2) w_ij < earth rotation timescale
            Since we only calculate one integer delay per scan, each scan should be < 0.4125 seconds to keep the delay from changing by more than 1/2 frame.
        3) w_ij > DM smearing timescale, 
            if we are using coherent dedispersion, this ensures that we have sufficient frequency resolution to upchannelize.
    
    """
    assert w_ij.shape == t_ij[0].shape == r_ij.shape
    iisort = np.argsort(t_ij[0,0,0,:]) # assume sorting of time windows is same for all stations, freq, and pointings
    # overlapping sub-integrations: 
    sub_scan_start = t_ij + 2.56e-6 * (w_ij // 2 - (w_ij * r_ij / 2))
    sub_scan_end = t_ij + 2.56e-6 * (w_ij // 2 + (w_ij * r_ij / 2))
    assert (sub_scan_end[:,:,:,iisort][:,:,:,0:-1] <= sub_scan_start[:,:,:,iisort][:,:,:,1:] + 2.56e-6).all(), "previous scan ends AFTER next one starts? you probably do not want this" # add 1 frame of buffer here to make this lenient
    
    # no changing integer lags
    earth_rotation_time = 0.4125  # seconds https://www.wolframalpha.com/input?i=1.28+us+*+c+%2F+%28earth+rotation+speed+*+2%29
    freq = np.linspace(800, 400, num=1024, endpoint=False)  # no nyquist freq
    assert np.max(w_ij * 2.56e-6) < earth_rotation_time, "Use smaller value of w_ij, scans are too long!"

    if dm is not None:  # check that wij exceeds smearing time
        dm_smear_sec = K_DM * dm * 0.390625 / freq**3
        diff = np.min(w_ij * 2.56e-6 - dm_smear_sec[:,None,None], axis=(-2,-1)) # check all frequency channels
        assert (diff > 0).all(), f"For DM = {dm}, w_ij needs to be increased by {-(np.min(diff)/ 2.56e-6):0.2f} frames to not clip the pulse within a channel" 
    return w_ij

def round_to_integer_frame(timestamps: np.ndarray, bbdata_filename):
    """Rounds to the integer frame number as specified by time0 in a given BBData.

    timestamps : np.ndarray of float64
        UNIX timestamps. Note that this is only precise to ~40 ns or so; the timestamps are stored at full precision in BBData.
    bbdata_filename : str
        For single baseband dumps, the filename out of which we should get the time0 to which we round.

    Returns
    -------
    timestamps_rounded : np.ndarray, same shape as :timestamps:
        UNIX timestamps which are an integer number of frames offset from time0 in bbdata_filename.
    """
    assert timestamps.shape[0] == 1024, "Only accepts full-band data for now, sorry!"
    if timestamps.ndim == 1:
        timestamps.shape = (1024,1)
    freq_id_present = get_all_im_freq(bbdata_filename)['id']
    time0_present = get_all_time0(bbdata_filename)
    time0_ctime_full = extrapolate_to_full_band(time0_present, freq_id_present)['ctime'][:]
    time0_ctime_full.shape = (1024,1)
    int_offset_full = np.rint((timestamps - time0_ctime_full) / 2.56e-6)
    """This gives absolute timestamps good to ~40 nanoseconds, limited by float64."""
    timestamps_rounded = time0_ctime_full + int_offset_full * 2.56e-6
    return timestamps_rounded.squeeze()

def fpga_start_time(
    time0, start_time_error=True, integer_error=False, astropy_time=True
):
    """Returns the FPGA start time, inferred from every channel of baseband data present, as a list of Decimal objects"""
    dump_time = Time(
        time0["ctime"][:],
        val2=time0["ctime_offset"][:],
        format="unix",
        precision=9,
    )
    fpga_start = dump_time - TimeDelta(2.56e-6 * un.s * time0["fpga_count"][:])
    fpga_start_unix = fpga_start.to_value("unix", subfmt="decimal")
    if (
        np.mod(np.median(fpga_start_unix), 1) >= 1e-9
    ):  # CHIME FPGAs start on an integer number of seconds , but not all F engines do
        if integer_error:
            ValueError("FPGAs do not start on an integer second")
        else:
            UserWarning("FPGAs do not start on an integer second")

    if (
        np.max(fpga_start_unix) - np.min(fpga_start_unix) > 1e-9
    ):  # should be within one FPGA cycle
        if start_time_error:
            ValueError("Frequency channel timestamps differ by more than 1 nanosecond!")
        else:
            UserWarning(
                "Frequency channel timestamps differ by more than 1 nanosecond!"
            )
    if astropy_time:
        return fpga_start
    else:
        return fpga_start_unix

def extrapolate_to_full_band(time0 : np.ndarray, freq_ids : np.ndarray):
    """Interpolates time0 to the full bandwidth to float64 precision, which is sufficient for specifying start times.

        Does this by:
        1) calculating the fpga_start_time with high precision, 
        2) doing nearest-neighbor interp on fpga_start_time, 
        3) then doing high-precision calculation of ctime and ctime_offset
        4) interpolating at the frequencies we do not have.

        This does something very similar to baseband_analysis.fill_waterfall()!

    """
    new_t0 = np.zeros((1024,), dtype = time0.dtype)
    new_fpga_start_time_unix = interp1d(
        x=freq_ids,
        y=fpga_start_time(time0, start_time_error = True, astropy_time = False).astype(float),
        kind="nearest",
        fill_value=(time0['fpga_count'][0],time0['fpga_count'][-1]),
        bounds_error=False,
    )  # Nearest neighbor interpolation for fpga_start_time. This should not really matter.
    missing = np.setdiff1d(np.arange(1024), freq_ids)

    # calculate new values for ctime and ctime_offset
    new_t0["fpga_count"] = np.round(
        np.interp(
            xp=freq_ids,
            fp=time0["fpga_count"],
            x=np.arange(1024),
            left=time0["fpga_count"][0],
            right=time0["fpga_count"][-1],
            )
        )  # FPGA count is always an integer!
    for freq_id, fpga_count, ftime in zip(
        np.arange(1024),
        new_t0["fpga_count"],
        new_fpga_start_time_unix(np.arange(1024)),
    ):
        if freq_id in missing:
            ct_decimal = Decimal(ftime) + fpga_count * Decimal(2.56e-6)
            # ...but store `ctime` and `ctime_offset` as floats
            new_t0["ctime"][freq_id] = ct_decimal
            new_t0["ctime_offset"][freq_id] = ct_decimal - Decimal(
                new_t0["ctime"][freq_id]
            )
            # calculate time0 and time0 offset

    # ...but keep original values where possible
    new_t0['ctime'][freq_ids] = time0['ctime']
    new_t0['ctime_offset'][freq_ids] = time0['ctime_offset']
    new_t0['fpga_count'][freq_ids] = time0['fpga_count']
    return new_t0

class CorrJob:
    def __init__(self, bbdata_filepaths, ras = None, decs = None):
        """Set up the correlation job:
        Get stations and order the bbdata_list as expected by difxcalc.
        Run difxcalc with a single pointing center.
        Choose station to use as the reference station, at which t_{ij} is initially inputted.
        For each station, define t_ij, w_ij, and r_ij into arrays of shape (N_baseline, N_freq, N_time) by calling define_scan_params().
        Given a set of BBData objects, define N * (N-1) / 2 baselines.
        Use run_difxcalc and save to self.calcresults so we only call difxcalc ONCE in the whole correlator job.
        """
        self.tel_names= []
        for path in bbdata_filepaths:
            self.tel_names.append(
                station_from_bbdata(
                    BBData.from_file(
                        path,
                        freq_sel = [0,-1],
                    )
                )
            )
        self.tel_names = sorted(self.tel_names) # sort tel_names alphabetically; this becomes the difxcalc antenna index mapping
        self.telescopes = [telescopes.tel_from_name(n) for n in self.tel_names]
        self.bbdata_filepaths = [bbdata_filepaths[ii] for ii in np.argsort(self.tel_names)] 
        bbdata_0 = BBData.from_file(bbdata_filepaths[0],freq_sel=[0,-1])
        # Get pointing centers from reference station, if needed.
        if ras is None:
            ras = bbdata_0['tiedbeam_locations']['ra'][:]
        if decs is None:
            decs = bbdata_0['tiedbeam_locations']['dec'][:]
        self.ras = np.atleast_1d(ras)
        self.decs = np.atleast_1d(decs)
        self.pointings = [
            ac.SkyCoord(ra=r * un.deg, dec=d * un.deg)
            for r, d in zip(self.ras.flatten(), self.decs.flatten())
        ]

        earliest_start_unix = np.inf
        latest_end_unix = -np.inf
        for filepath in bbdata_filepaths:
            this_bbdata = BBData.from_file(filepath,freq_sel = [0,-1])
            assert same_pointing(bbdata_0,this_bbdata)
            earliest_start_unix = min(earliest_start_unix,
                this_bbdata['time0']['ctime'][0])
            latest_end_unix = max(latest_end_unix, 
                this_bbdata['time0']['ctime'][-1] + this_bbdata.ntime)

        earliest_start_unix = int(earliest_start_unix - 1) # buffer
        duration_sec = int(latest_end_unix - earliest_start_unix + 1.0 )
        calcfile_name = os.path.join(CALCFILE_DIR, 'pyfx_corrjob_' + str(bbdata_0.attrs['event_id']) + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.calc')
        _ = io.make_calc(
            telescopes =self.telescopes,
            sources = self.pointings,
            time = Time(earliest_start_unix, format = 'unix', precision = 9),
            duration_sec=duration_sec,
            ofile_name=calcfile_name,
        )
        self.calcresults = runner.run_difxcalc(
            calcfile_name,
            sources=self.pointings,
            telescopes=self.telescopes,
            force=True,
            remove_calcfile=True,
            difxcalc_cmd=DIFXCALC_CMD,
        )
        return 

        #for ii_b in range(len(self.tel_names)):
        #tau_00 = self.calcresults.baseline_delay(
        #    ant1=self.tel_names.index(ref_station),
        #    ant2=ii_b,
        #    time=Time(unix_a, format="unix"),
        #    src=src,  # should be a 1 time value x 1 pointing evaluation
        #)

    def t0_f0_from_bbdata_filename(self,t0f0,bbdata_ref_filename):
        """ Returns the actual t00 and f0 from bbdata_ref.

        This allows you to specify, e.g. the "start" of the dump at the "top" of the band by passing in ("start","top").

        """

        (_t0, _f0) = t0f0
        if _f0 == 'top': # use top of the collected band as reference freq
            iifreq = 0
        if _f0 == 'bottom': # use bottom of the collected band as reference freq
            iifreq = -1
        im_freq = get_all_im_freq(bbdata_ref_filename)
        time0 = get_all_time0(bbdata_ref_filename)
        if type(_f0) is float: # use the number as the reference freq
            iifreq = np.argmin(np.abs(im_freq['freq']['centre'][:] - _f0))
            offset_mhz = im_freq["centre"][iifreq] - _f0
            print('INFO: Offset between requested frequency and closest frequency: {offset_mhz} MHz')
        f0 = im_freq['centre'][iifreq] # the actual reference frequency.

        if _t0 == 'start':
            t0 = time0['ctime'][iifreq]  # the actual reference start time
        if _t0 == 'middle':
            ntime = get_ntime(bbdata_ref_filename)
            t0= time0['ctime'][iifreq] + ntime * 2.56e-6 // 2
        return t0,f0
        
    def define_scan_params(
        self,
        ref_station = 'chime',
        t0f0 = ('start','top'),
        start_or_toa = 'start',
        time_spacing = 'even',
        freq_offset_mode = 'bbdata',
        Window = np.ones(1024,dtype = int) * 1000,
        r_ij = 1.0,
        period_frames = None,
        num_scans_before = 10,
        num_scans_after = 10,
        time_ordered = False,
        pdot = 0,
        dm = None,
        max_lag = 20,
    ):
        """
        Tells the correlator when to start integrating, how long to start integrating, for each station. Run this after the CorrJob() is instantiated.

        ref_station : station name
            For example, use 'chime'. Could also reference to others.
        t0f0 : tuple consiting of two floats
            First float: denotes the start time of the integration, either with a unix time or a keyword ('start' or 'middle'). 
            Second float: denotes frequency channel. Can also pass keywords (either 'top' or 'bottom') to indicate the top or bottom of the available frequency band.
        start_or_toa : 'start' or 'toa'
            Interpret the given time as a start time or a "center" time.
        time_spacing : 'even', or 'p+pdot'
            Equal number of gates
        
        freq_offset_mode : 'bbdata', or 'dm'
        
        width : 'fixed', 'from_time'
        
        Window : np.ndarray
            Sets the integration duration in frames as a function of frequency.
        dm : float
            A dispersion measure for gating. Get this right to the 2-3rd decimal place.
        
        kwargs : 'dm' and 'f0', 'period_frames', 'pdot','wi'
        """
        bbdata_ref_filename = self.bbdata_filepaths[self.tel_names.index(ref_station)]
        # First: if t0f0 is a bunch of strings, then get t0 & f0 from the BBData. 
        if type(t0f0[0]) is not float and type(t0f0[1]) is not float:
            t00, f0 = self.t0_f0_from_bbdata_filename(t0f0, bbdata_ref_filename)
        else: # OK, I guess we were given t00 and f0
            (t00, f0) = t0f0

        # First do t_i0 for the reference station...
        if freq_offset_mode == "bbdata":
            _ti0 = _ti0_from_t00_bbdata(bbdata_ref_filename, t_00 = t00, f0 = f0) 
        if freq_offset_mode == "dm":
            _ti0 = _ti0_from_t00_dm(bbdata_ref_filename, t00, f0, dm = dm, fi = FREQ)

        # If _ti0 is a TOA, need to shift _ti0 back by half a scan length 
        if start_or_toa == 'toa':
            _ti0 -= Window *2.56e-6 / 2  # Scan duration given by :Window:, not :period_frames:!
            print('INFO: Using TOA mode: shifting all scans to be centered on _ti0')

        assert Window.shape[0] == 1024, "Need to pass in the length of the integration as a function of frequency channel!"
        if period_frames is None:
            period_frames = Window

        # Next do t_ij for the reference station from t_i0.
        period_frames = np.atleast_1d(period_frames)
        if period_frames.size == 1:
            period_frames = np.zeros(1024) + period_frames

        # Allow evenly spaced gates...
        if time_spacing == "even":
            _tij = _ti0_even_spacing(bbdata_ref_filename,_ti0,period_frames,num_scans_before = num_scans_before, num_scans_after = num_scans_after, time_ordered = time_ordered)
        if time_spacing == "p+pdot": #...or start times that get later and later (pdot).
            _tij = _ti0_ppdot(_ti0, period_i, bbdata_a, bbdata_b)

        print('Success: generated a valid set of integrations! Now call run_correlator_job() or run_correlator_job_multiprocessing()')
        t_ij_station_pointing = self.tij_other_stations(_tij, ref_station = ref_station)
        
        # Check that the time spacing works.
        if Window.ndim == 1: # broadcast to the shape of tij
            Window = Window[:,None,None] + np.zeros_like(t_ij_station_pointing[0])
        if r_ij.ndim == 1: # broadcast to the shape of tij
            r_ij = r_ij[:,None,None] + np.zeros_like(t_ij_station_pointing[0])
        
        validate_wij(Window,t_ij_station_pointing, r_ij, dm = dm)

        self.max_lag = max_lag
        return t_ij_station_pointing,  Window.astype(int), r_ij

    def tij_other_stations(self, tij, ref_station = 'chime'):
        """Do this on a per-pointing and per-station basis."""
        iiref = self.tel_names.index(ref_station)
        tij_sp = np.zeros(
            (len(self.telescopes),
             1024,
            len(self.pointings),
            tij.shape[-1]
            ),
            dtype = float) 
            # tij_sp.shape = (n_station, n_freq, n_pointing, n_time)

        for iitel, telescope in enumerate(self.telescopes):
            for jjpointing, pointing in enumerate(self.pointings):
                tau_ij = self.calcresults.baseline_delay(
                    ant1 = iiref, 
                    ant2 = iitel,
                    time = Time(tij.flatten(),format = 'unix'),
                    src = jjpointing,
                    ).reshape(tij.shape)
                tij_sp[iitel,:,jjpointing,:] = tij + tau_ij
        return tij_sp

    def visualize_twr(self,bbdata_ref_filename,t,w,r,pointing = 0,dm = None,fscrunch = 4, tscrunch = None):
        bbdata_A = BBData.from_file(bbdata_ref_filename)
        iiref = self.tel_names.index(station_from_bbdata(bbdata_A))
        fill_waterfall(bbdata_A,write = True)
        if dm is not None:
            from baseband_analysis.core.dedispersion import coherent_dedisp
            # TODO: CL: need to replace with pyfx.core_correlation.intrachannel_dedisp
            wfall = coherent_dedisp(data = bbdata_A,DM = dm)
        else:
            wfall = bbdata_A['tiedbeam_baseband'][:]
        wwfall = np.abs(wfall)**2
        wwfall -= np.median(wwfall,axis = -1)[:,:,None]
        wwfall /= median_abs_deviation(wwfall,axis = -1)[:,:,None]
        if tscrunch is None:
            tscrunch = int(np.median(w) // 10 )
        sww = _scrunch(wwfall,fscrunch = fscrunch, tscrunch = tscrunch)
        del wwfall
        f = plt.figure()
        plt.imshow(sww[:,pointing] + sww[:,pointing+1],aspect = 'auto',vmin = -1,vmax = 3)
        del bbdata_A

        y = np.arange(1024)
        for iiscan in range(t.shape[-1]):
            x_start = (t[iiref,:,pointing,iiscan] - bbdata_A['time0']['ctime'][:]) / (2.56e-6 * tscrunch)
            x_end = x_start + w[:,pointing,iiscan] / tscrunch
            x_mid = x_start + (x_end - x_start) * 0.5 
            x_rminus = x_mid - (x_end - x_start) * 0.5 * r[:,pointing,iiscan]
            x_rplus = x_mid + (x_end - x_start) * 0.5 * r[:,pointing,iiscan]
            plt.fill_betweenx(x1 = x_start, x2 = x_end,y = y/fscrunch,alpha = 0.15)
            if iiscan == 0:
                linestyle = '-'
            else:
                linestyle = '--'
            plt.plot(x_start, y/fscrunch,linestyle = linestyle,color = 'black')
            plt.plot(x_end, y/fscrunch,linestyle = linestyle,color = 'black')
        xmin = np.min(t[iiref,:,pointing,:] - bbdata_A['time0']['ctime'][:,None],axis = -1) / (2.56e-6 * tscrunch)
        xmax = np.max(t[iiref,:,pointing,:] - bbdata_A['time0']['ctime'][:,None],axis = -1) / (2.56e-6 * tscrunch)

        plt.xlim(np.median(xmin),np.median(xmax))
        plt.ylim(1024 / fscrunch,0)
        plt.ylabel('Freq ID')
        plt.xlabel(f'Time ({tscrunch:0.1f} frames)')
        return f

    def run_correlator_job(self,t_ij, w_ij, r_ij, dm = None, event_id = None, out_h5_file = None):
        """Run auto- and cross- correlations.

        Loops over baselines, then frequencies, which are all read in at once and ordered using fill_waterfall. This works well on short baseband dumps. 
        Memory cost: 2 x BBData, 
        I/O cost:N*(N-1) / 2 x BBData.

        Parameters
        ----------
        t_ij : np.ndarray
            Of start times as a function of (n_station, n_freq, n_pointing, n_time)
        w_ij : np.ndarray
            Of start times as a function of (n_pointing, n_time)
        r_ij : np.ndarray
            Of start times as a function of (n_freq, n_pointing, n_time)
        dm : float
            A dispersion measure for de-smearing. Fractional precision needs to be 10%.
        """
        output = VLBIVis()
        pointing_centers = np.zeros((len(self.pointings),),dtype = output._dataset_dtypes['pointing'])
        pointing_centers['corr_ra'] = self.ras
        pointing_centers['corr_dec'] = self.decs
        for iia in range(len(self.tel_names)):
            bbdata_a = BBData.from_file(self.bbdata_filepaths[iia])
            fill_waterfall(bbdata_a, write = True)
            indices_a = np.round((t_ij[iia] - bbdata_a['time0']['ctime'][:,None,None]) / 2.56e-6).astype(int) # shape is (nfreq, npointing, nscan)
            # there are scans with missing data: check the start and end index
            mask_a = (indices_a < 0) + (indices_a  + w_ij[None,:,:] > bbdata_a.ntime) 
            indices_a[mask_a] = int(bbdata_a.ntime // 2)
            # ...but we just let the correlator correlate
            auto_vis = autocorr_core(dm, bbdata_a, 
                                    t_a = indices_a,
                                    window = w_ij,
                                    R = r_ij,
                                    max_lag = self.max_lag, 
                                    n_pol = 2)
            # ...and replace with nans afterward.
            auto_vis += mask_a[:,:,None,None,None,:] * np.nan # fill with nans where
            int_time = np.zeros(shape = t_ij[iia].shape, dtype = output._dataset_dtypes['time'])
            int_time["ctime"][:] = t_ij[iia]
            int_time["duration_frames"][:] = w_ij
            int_time["dur_ratio"][:] = r_ij
            int_time["on_window"][:] = True
            print('TODO: Specify on window in define_scan_params according to source type')

            output._from_ndarray_station(
                event_id,
                telescope = self.telescopes[iia],
                bbdata = bbdata_a,
                auto = auto_vis,
                integration_time = int_time,
                )

            for iib in range(iia+1,len(self.tel_names)):
                bbdata_b = BBData.from_file(self.bbdata_filepaths[iib])
                fill_waterfall(bbdata_b, write = True)
                vis = crosscorr_core(
                        bbdata_a, 
                        bbdata_b, 
                        indices_a,
                        w_ij, 
                        r_ij,
                        self.calcresults, 
                        DM = dm, 
                        index_A = iia,
                        index_B = iib,
                        max_lag = self.max_lag, 
                        complex_conjugate_convention = -1, 
                        intra_channel_sign = 1
                    )
                output._from_ndarray_baseline(
                        event_id = event_id,
                        pointing_center = pointing_centers,
                        telescope_a = self.telescopes[iia],
                        telescope_b = self.telescopes[iib],
                        cross = vis,
                    )
                del bbdata_b # free up space in memory
            del bbdata_a

        if type(out_h5_file) is str:
            output.save(out_h5_file)
        return output

    def run_correlator_job_one_freq_id(t_ij, w_ij, r_ij, dm, event_id = None,):
        """Loops over baselines, then frequencies, which are read in one by one. This works on tracking beams."""

    def run_correlator_job_multiprocessing(t_ij, w_ij, r_ij):
        """
        output = VLBIVis()
        for b in baselines:
            for f in freq:
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
        """
        return NotImplementedError("Left as an exercise to the reader.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Module for doing VLBI. In a python script, you can run:
        job = CalcJob()

        # steady source: Use a duty cycle of 1 (width = 1) and integrate over the full gate (subwidth = 1)
        t_ij, w_ij, r_ij = job.define_scan_params(t00: unix_time, period_i = 0.005 sec, freq_offset_mode = 'bbdata', time_spacing = 'period', dm = 500)

        # pulsar or FRB: Set the period to the pulsar period, use frequency offset based on dispersion measure of the pulsar, and integrate over a small fraction of the full gate (subwidth = 0.2):
        t_ij, w_ij, r_ij = job.define_scan_params(t00: unix_time, period_i = 0.005 sec, freq_offset_mode = 'dm', time_spacing = 'period', dm = 500, width = 1, subwidth = 0.5)

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
        help='How to calculate frequency offset? (Options: "bbdata", "dm")',
        default="bbdata",
    )
    parser.add_argument(
        "out_file",
        help="E.g. /path/to/output_vis.h5",
        default='./vis.h5',
    )
    cmdargs = parser.parse_args()
    job = CorrelatorJob(bbdata_filepaths, ref_station = 'chime',ras = cmdargs.ra, decs = cmdargs.dec)
    t_ij, w_ij, r_ij = job.define_scan_params(
        t00=cmdargs.t00,  # unix
        period_i=0.005,  # seconds
        freq_offset_mode=cmdargs.time_offset,
        time_spacing=cmdargs.freq_offset,
        dm=500,  # pc cm-3
    )
    if parallel:
        output = run_correlator_job_multiprocessing(t_ij, w_ij, r_ij)
    else:
        output = run_correlator_job(t_ij, w_ij, r_ij)
    output.save(parser.out_file)