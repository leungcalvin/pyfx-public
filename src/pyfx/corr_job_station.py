import os,datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import astropy
import astropy.coordinates as ac
import astropy.units as un
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import fill_waterfall,_scrunch
from pycalc11 import Calc
from scipy.fft import fft, ifft, next_fast_len
from scipy.stats import median_abs_deviation
from decimal import Decimal
from astropy.time import Time,TimeDelta
import logging
from pyfx.core_correlation_station import autocorr_core, cross_correlate_baselines
from pyfx.bbdata_io import station_from_bbdata, get_all_time0, get_all_im_freq
from coda.core import VLBIVis
from typing import List, Optional,Tuple
"""Tools for extracting good values of t,w,r for correlator gating.
The main functions provided are get_twr_continuum and get_twr_singlepulse"""

import numpy as np 

def first_valid_frame(arr, axis, invalid_val=0):
    """Gets first valid element as a function of frequency axis.
    arr : np.array 2d
        Pass in np.isfinite(bbdata['tiedbeam_baseband'] as arr
    
    axis : int
        To look for first nonzero element along e.g. the last axis pass in axis = -1.
    
    invalid_val : int
        If no nonzero (i.e. invalid) elements,
    """
    mask = np.isfinite(arr).astype(bool)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def duration_frames(arr,axis = -1):
    """Gets number of valid samples counted over the specified axis."""
    return np.sum(np.isfinite(arr),axis = axis)

def tw2twr_frames(
    t:np.ndarray,w):
    """Converts from freq dependent w to freq dependent r. 
    Here t is assumed to be an index; therefore it is quantized to integers.
    Up to rounding errors in w, this is the inverse of tw2twr()."""
    r_new = w / np.max(w) # one per frequency
    w_new = np.max(w) # one per band
    return t.astype(int), w_new.astype(int), r_new

def twr2tw_frames(t,w,r):
    """Converts from freq dependent r to freq dependent w. 
    Here t is assumed to be an index; therefore it is quantized to intege
    Up to rounding errors in w, this is the inverse of tw2twr().
    """
    return t.astype(int), (w * r).astype(int)

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
FREQ = np.linspace(800,400,num = 1024, endpoint = False)

def right_broadcasting(arr, target_shape):
    return arr.reshape(arr.shape + (1,) * (len(target_shape) - arr.ndim))

def same_pointing(bbdata_a,bbdata_b):
    assert np.isclose(bbdata_a['tiedbeam_locations']['ra'][:],bbdata_b['tiedbeam_locations']['ra'][:]).all()
    assert np.isclose(bbdata_a['tiedbeam_locations']['dec'][:],bbdata_b['tiedbeam_locations']['dec'][:]).all()
    return True

def validate_wij(w_ij, r_ij, dm):
    """Performs some following checks on w_ij
    
    Parameters
    ----------
    w_ij : np.ndarray of ints
        Integration length, in frames, as a function of (n_freq, n_pointing, n_time)

    t_ij : np.ndarray of float64
        UNIX time (in seconds) at which integration starts @ each station (n_station, n_freq, n_pointing, n_time)

    r_ij : np.ndarray of float64
        A number between 0 and 1 denoting the sub-integration, (n_freq, n_pointing, n_time).

    dm : np.ndarray of float64 of shape (n_pointing)

    Returns
    -------
    True : If all the following checks pass...
        1) No overlapping sub-integrations (integrations might overlap, if the smearing timescale within a channel exceeds the pulse period
        2) w_ij < earth rotation timescale
            Since we only calculate one integer delay per scan, each scan should be < 0.4125 seconds to keep the delay from changing by more than 1/2 frame.
        3) w_ij > DM smearing timescale, 
            if we are using coherent dedispersion, this ensures that we have sufficient frequency resolution to upchannelize.
    
    """
    # no changing integer lags
    earth_rotation_time = 0.4125  # seconds https://www.wolframalpha.com/input?i=1.28+us+*+c+%2F+%28earth+rotation+speed+*+2%29
    freq = np.linspace(800, 400, num=1024, endpoint=False)  # no nyquist freq
    assert np.max(w_ij * 2.56e-6) < earth_rotation_time, "Use smaller value of w_ij, scans are too long!"

    if dm is not None:  # check that wij exceeds smearing time for that pointing
        dm_smear_sec = K_DM * dm[None,:,None] * 0.390625 / freq[:,None,None]**3
        diff = np.min(w_ij * 2.56e-6 - dm_smear_sec, axis=(-2,-1)) # check all frequency channels
        if not (diff > 0).all():
            logging.warning(f"For DM = {dm}, w_ij needs to be increased by {-(np.min(diff)/ 2.56e-6):0.2f} frames to not clip the pulse within a channel")
    return w_ij

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

def initialize_pointing_from_coords(ras, decs,names, dm_trials):
    assert ras.shape == decs.shape
    assert names.shape == dm_trials.shape
    assert ras.size == names.size * 2
    pointing_spec = np.empty(names.shape, dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = ras[::2]
    pointing_spec['corr_dec'][:] = decs[::2]
    pointing_spec['source_name'][:] = names[:]
    pointing_spec['corr_dm'][:] = dm_trials[:]
    return pointing_spec

def initialize_pointing_dm_trials(single_ra, single_dec, single_name, dm_trials):
    assert ras.shape == decs.shape
    assert names.shape == dm_trials.shape
    assert ras.size == names.size * 2
    pointing_spec = np.empty(names.shape, dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = single_ra
    pointing_spec['corr_dec'][:] = single_dec
    pointing_spec['source_name'][:] = np.array([single_name + f'_DMT_{ii:0>3d}' for ii in range(len(dm_trials))])
    pointing_spec['corr_dm'][:] = dm_trials[:]
    return pointing_spec

def atime2ctimeo(at):
    """Convert astropy.Time into ctime & ctime_offset"""
    ctime_long = at.to_value('unix', 'long') # convert to float128
    ctime = ctime_long.astype(np.float64) # break into float64
    ctime_offset = (ctime_long - ctime).astype(np.float64) # take the remainder as another float64
    return ctime,ctime_offset

def ctimeo2atime(ctime,ctime_offset):
    """Convert ctime & ctime_offset into astropy.Time"""
    return Time(val = ctime, val2 = ctime_offset,format = 'unix')

class CorrJob:
    def __init__(
        self, 
        pointing_spec : np.ndarray,
        bbdatas: List[BBData],
        telescopes:List[ac.earth.EarthLocation], 
        bbdata_filepaths:Optional[List[str]]=None, 
        ref_station:Optional[str]='chime',
        default_max_lag=100):
        """Set up the correlation job. 

        __init__ initializes the pointing map and the pointings, and runs difxcalc.

        Get stations and order the bbdata_list as expected by difxcalc.
        Run difxcalc with a single pointing center.
        Choose station to use as the reference station, at which t_{ij} is initially inputted.
        For each station, define t_ij, w_ij, and r_ij into arrays of shape (N_baseline, N_freq, N_time) by calling define_scan_params().
        Given a set of BBData objects, define N * (N-1) / 2 baselines.
        Use run_difxcalc and save to self.pycalc_results so we only call difxcalc ONCE in the whole correlator job.
        """
        self.tel_names = [telescopes[i].info.name for i in range(len(telescopes))]
        self.bbdatas = bbdatas
        # get tel names
        for i,this_bbdata in enumerate(bbdatas):
            tel_name=station_from_bbdata(
                    this_bbdata,
                    )
            if tel_name != self.tel_names[i]:
                print(f"warning: telescope name { self.tel_names[i]} from input telescopes does not correspond to telescope name {tel_name} from bbdata. Please check that your input parameters are identically ordered.")
        self.telescopes = telescopes 
        self.ref_station=ref_station
        ref_index=self.tel_names.index(ref_station)
        self.ref_index=ref_index
        bbdata_ref = self.bbdatas[self.ref_index]

        earliest_start_unix = np.inf
        latest_end_unix = -np.inf

        # loop again to check data format specifications
        for i,this_bbdata in enumerate(bbdatas):
            assert same_pointing(bbdata_ref,this_bbdata)
            earliest_start_unix = min(earliest_start_unix,
                this_bbdata['time0']['ctime'][0])
            latest_end_unix = max(latest_end_unix, 
                this_bbdata['time0']['ctime'][-1] + this_bbdata.ntime)
            if i==ref_index:
                self.ref_ctimes=this_bbdata['time0']['ctime']
                self.ref_ctime_offsets=this_bbdata['time0']['ctime_offset']
            if this_bbdata.nfreq<1024:
                fill_waterfall(this_bbdata, write=True)

        earliest_start_unix = int(earliest_start_unix - 1) # buffer
        duration_min = 3 #max(int(np.ceil(int(latest_end_unix - earliest_start_unix + 1.0 )/60)),1)
        self.pointings = pointing_spec
        self.pointings_sc = ac.SkyCoord(ra=pointing_spec['corr_ra'].flatten() * un.deg, 
                                        dec=pointing_spec['corr_dec'].flatten() * un.deg, 
                                        frame='icrs')
        duration_min=1
        ci = Calc(
                station_names=self.tel_names,
                station_coords=self.telescopes,
                source_coords=self.pointings_sc,
                start_time=Time(np.floor(earliest_start_unix), format = 'unix', precision = 9),
                duration_min=duration_min,
                base_mode='geocenter', 
                dry_atm=True, 
                wet_atm=True,
                d_interval=1,
            )
        ci.run_driver()
        self.pycalc_results=ci
        self.max_lag = default_max_lag #default value if define_scan_params is not called (continuum sources)
        return 

    def t0_f0_from_bbdata(self,
        t0f0:Tuple[str],
        bbdata_ref,
        return_ctimeo = False
        ):
        """ Returns the actual t00 and f0 from bbdata_ref, which could be either a BBData object or a filepath.

        This allows you to specify, e.g. the "start" of the dump at the "top" of the band by passing in ("start","top").
        This function bypasses memh5 and uses hdf5 directly to peek at the files without consuming much additional RAM. 

        Parameters
        ----------
        t0f0 : tuple 
            Specifying start/middle of the dump in time and top/bottom of the frequency band
        bbdata_ref : name of filename, or BBData

        Returns
        -------
        t0 : astropy.Time
            Specifying t0.
        f0 : float
            Specifying reference frequency in MHz
        """
        bbdata_ref = self.bbdatas[self.ref_index]
        
        if type(bbdata_ref) is BBData:
            im_freq = bbdata_ref.index_map['freq']
            time0 = bbdata_ref['time0']
            ntime = bbdata_ref.ntime
        elif type(bbdata_ref) is str:
            im_freq = get_all_im_freq(bbdata_ref)
            time0 = get_all_time0(bbdata_ref)
            ntime = get_ntime(bbdata_ref)
 
        (_t0, _f0) = t0f0
        if _f0 == 'top': # use top of the collected band as reference freq
            iifreq = 0
        if _f0 == 'bottom': # use bottom of the collected band as reference freq
            iifreq = -1
        if type(_f0) is float: # use the number as the reference freq
            iifreq = np.argmin(np.abs(im_freq['centre'][:] - _f0))
            offset_mhz = im_freq["centre"][iifreq] - _f0
            logging.info('INFO: Offset between requested frequency and closest frequency: {offset_mhz} MHz')
        f0 = im_freq['centre'][iifreq] # the actual reference frequency in MHz.

        if _t0 == 'start':
            t0 = Time(time0['ctime'][iifreq],val2 = time0['ctime_offset'][iifreq]) # the actual reference start time
        if _t0 == 'middle':
            t0 = Time(time0['ctime'][iifreq] + ntime * 2.56e-6 // 2,
                    val2 = time0['ctime_offset'][iifreq])
        else:
            t0 = _t0
        return t0,f0

    def define_scan_params_continuum(
        self,
        equal_duration = True, 
        pad = 100):
        """Return start frame and number of window frames ("t" and "w") by looking at the nan pattern in telA_bbdata, making use of ~all the data we have.
        
        This is an appropriate way to get the t,w,r data for a single phase-center pointing on a continuum source.

        To take into account that the different boundaries of the data, we trim the edges of the scan.
        We remove :pad: frames from both the left and right of the integration.gate_start_frame & ww.

        Inputs
        ------
        self : CorrJob
            Should have self.bbdatas and self.ref_index attributes.
        
        equal_duration : bool
            If equal_duration is set to True, we will make sure the integration time is the same across all frequencies.
        """
        bbdata_ref = self.bbdatas[self.ref_index]
        pol=0
        n_channels = 1024
        wfall_data = bbdata_ref['tiedbeam_baseband'][:,pol,:] # both polarizations should have same nan pattern because Kotekan works in freq  time element order.
        assert wfall_data.shape[0] == n_channels, f"Frequency channels missing, please pass in {n_channels} channels"
        gate_start_frame =  first_valid_frame(wfall_data,axis = -1)
        window = duration_frames(wfall_data,axis = -1)
        if wfall_data.shape[-1] // 2 > np.median(window):
            logging.info(f'WARNING: More than half of the channels invalid in data for {telA_bbdata.attrs["event_id"]}')
        if equal_duration: # choose the length that maximizes the sensitivity as quantified by num_valid_channels x duration
            sorted_window_lengths = np.sort(window)
            sens_metric = sorted_window_lengths * (1024 - np.arange(1024)) # n_time * n_channels with that many valid samples
            window = sorted_window_lengths[np.argmax(sens_metric)] + np.zeros(1024,dtype=int)
        assert (np.min(window) - 2 * pad > 0), "twr params result in negative integration duration when zero-padded. Please optimize manually, e.g. decrease pad value or input twr manually."
        tt,ww,rr =tw2twr_frames(gate_start_frame + pad, window - 2 * pad)
        ww = np.atleast_2d(ww)
        tt.shape = (1024,len(self.pointings),1)
        rr.shape = (1024,len(self.pointings),1)

        gate_spec = np.empty(tt.shape,dtype = VLBIVis._dataset_dtypes['time'])
        gate_spec['gate_start_unix'], gate_spec['gate_start_unix_offset'] = atime2ctimeo(self.frame2atime(tt))
        gate_spec['gate_start_frame'] = tt
        gate_spec['window'] = ww
        gate_spec['r_ij'] = 1.0
        return gate_spec
        
    def define_scan_params_transient(
        self,
        t0f0:Optional[Tuple[float,float]]=None,
        start_or_toa = 'start',
        freq_offset_mode = 'bbdata',
        time_spacing = 'even',
        window = np.ones(1024,dtype = int) * 1000,
        r_ij = np.ones(1024),
        num_scans_before = 10,
        num_scans_after = 8,
        time_ordered = False,
        period_frames = 1000,
    ):
        """
        Tells the correlator when to start integrating, how long to start integrating, for each station. Run this after the CorrJob() is instantiated.

        t0f0 : tuple consiting of two floats
            First float: denotes the start time of the integration with a float (UNIX time), or 'start' or 'middle'
            Second float: denotes frequency channel in MHz, or 'top' or 'bottom'.
        start_or_toa : 'start' or 'toa'
            Interpret the given time as a start time or a "center" time.

        freq_offset_mode : 'bbdata', or 'dm'
            If freq_offset_mode == 'bbdata', get the gating from the dataset start times.
            If freq_offset_mode == 'dm', get the gating DM from the pointing spec.
                In principle it is legal and slightly more flexible and slightly more sensitive 
                to allow for a gate DM (incoherent DM) which differs from the coherent DM.
                However, let's not allow this to reduce confusion.
        
        width : 'fixed', 'from_time'
        
        Window : np.ndarray of int
            Sets the integration duration in frames as a function of frequency.

        kwargs : 'dm' and 'f0', 'pdot','wi'
        """
        dm = self.pointings['dm_correlator'] 
        bbdata_ref = self.bbdatas[self.ref_index]
        window = np.atleast_1d(window)
        assert np.issubdtype(window.dtype, np.integer),'Window must be an integer number of frames!'

        # First: if t0f0 is a bunch of strings, then get t0 & f0 from the BBData. 
        # t00 will be output as an astropy.Time
        # f0 will be output as a float
        t00,f0 = self.t0_f0_from_bbdata(t0f0,bbdata_ref = bbdata_ref)
        
        # First do t_i0 for the reference station, i.e. generate start times for other frequencies.
        if freq_offset_mode == 'bbdata':
            _ti0 = _ti0_from_t00_bbdata(bbdata_ref,t_00 = t00, f0 = f0,return_ctimeo = False)
        if freq_offset_mode == "dm":
            assert type(t0f0[0]) is not str and type(t0f0[1]) is not str, "You probably want to pass in a hard-coded time & frequency reference as t0f0 if you want to follow a DM sweep."
            t_i0 = self._ti0_from_t00_dm(t00, f0, dm = dm, fi = FREQ,return_ctimeo=False) # frame indices
        # If _ti0 is a TOA, need to shift _ti0 back by half a scan length 
        
        window = right_broadcasting(window,target_shape = t_i0.shape)
        if start_or_toa == 'toa':
            t_i0 -= window # Scan duration given by :Window:, not :period_frames:!
            logging.info('INFO: Using TOA mode: shifting all scans to be centered on t_ij')

        # Next do t_ij for the reference station from t_i0.
        if np.abs(num_scans_after + num_scans_before) >= 1:
            period_frames = np.atleast_1d(period_frames)
            if period_frames.size == 1:
                period_frames.shape = (len(self.pointings),)

        # Allow evenly spaced gates...
        if time_spacing == "even":
            _tij = self._ti0_even_spacing(t_i0,period_frames,
            num_scans_before = num_scans_before, 
            num_scans_after = num_scans_after, 
            time_ordered = time_ordered, return_ctimeo = False)
        if time_spacing == "overlap2":
            _tij1 = self._ti0_even_spacing(t_i0,period_frames,
                                num_scans_before = num_scans_before, 
                                num_scans_after = num_scans_after, 
                                time_ordered = time_ordered, return_ctimeo = False)
            _tij2 = self._ti0_even_spacing(t_i0 + 0.5 * period_frames,period_frames,
                                num_scans_before = num_scans_before, 
                                num_scans_after = num_scans_after, 
                                time_ordered = time_ordered, return_ctimeo = False)
            _tij = np.concatenate((_tij1,_tij2),axis = -1) # concatenate along time axis
        if time_spacing == 'overlap3':
            _tij1 = _ti0_even_spacing(bbdata_ref,
                                t_i0,period_frames,
                                num_scans_before = num_scans_before, 
                                num_scans_after = num_scans_after, 
                                time_ordered = time_ordered,
                                return_ctimeo = False)
            _tij2 = _ti0_even_spacing(bbdata_ref,
                                t_i0 + 0.333 * period_frames,period_frames,
                                num_scans_before = num_scans_before, 
                                num_scans_after = num_scans_after, 
                                time_ordered = time_ordered,
                                return_ctimeo = False)
            _tij3 = _ti0_even_spacing(bbdata_ref,
                                t_i0 + 0.666 * period_frames,period_frames,
                                num_scans_before = num_scans_before, 
                                num_scans_after = num_scans_after, 
                                time_ordered = time_ordered,
                                return_ctimeo = False)
            _tij = np.concatenate((_tij1,_tij2,_tij3),axis = -1) # concatenate along time axis

        logging.info('Success: generated a valid set of integrations! Rounding to nearest 2.56us to create topocentric gate specification.')
        t_ij = self.frame2atime(int_frames = _tij)
        window = np.broadcast_to(right_broadcasting(window,target_shape = t_ij.shape),t_ij.shape)
        gate_spec = np.empty(_tij.shape,dtype = VLBIVis._dataset_dtypes['time'])
        gate_spec['gate_start_unix'], gate_spec['gate_start_unix_offset'] = atime2ctimeo(t_ij)
        gate_spec['gate_start_frame'] = _tij
        gate_spec['duration_frames'] = window + np.zeros_like(t_ij)
        gate_spec['dur_ratio'] = r_ij[:,None,None]
        validate_wij(gate_spec['duration_frames'], r_ij, dm = dm) # check DM smearing not too large
        return gate_spec

    def atime2frame(
        self,
        timestamps: np.ndarray):
        """Convert astropy.Time to frame index using self.ref_bbdata"""        
        ctime = Time(
                    self.ref_ctimes,
                    val2=self.ref_ctime_offsets,
                    format="unix",
                    precision=9,
                )
        _ctime = right_broadcasting(ctime, target_shape = timestamps.shape)
        closest_frame=np.round(((timestamps - _ctime).sec/(2.56e-6*un.s)).value).astype(int)
        return closest_frame #timestamps_rounded

    def frame2atime(
        self,
        int_frames: np.ndarray):
        """Convert frame index to astropy.Time"""
        ctime = Time(
            right_broadcasting(self.ref_ctimes,int_frames.shape) + 2.56e-6 * int_frames,
            val2 = right_broadcasting(self.ref_ctime_offsets,int_frames.shape),
            format = 'unix',
            precision = 9,
        )
        return ctime
    
    def _ti0_from_t00_bbdata(bbdata, t_00,f0,return_ctimeo = False):
        """Returns ti0 such that it aligns with the start frame of each channel in the bbdata (potentially with a constant frame_offset > 0 for all channels).

        This always returns 1024 start times, since different baselines might have different amounts of frequency coverage.

        Parameters
        ----------
        bbdata_a : A BBData object used to define the scan start

        frame_offset : int
            An integer number of frames (must be non0-negative) to skip.

        return_ctimeo : bool
            If True, returns ctime, ctime_offset
            If False, returns integer frame numbers.
        Returns
        -------
        ti0 : np.array of shape (1024,) of true times.
        """
        if type(bbdata) is str:
            im_freq = get_all_im_freq(bbdata_filename)
            t0 = get_all_time0(bbdata_filename)
        elif type(bbdata) is BBData:
            im_freq = bbdata.index_map['freq']
            t0 = bbdata['time0'][:].copy()
        iifreq = np.argmin(np.abs(im_freq["centre"][:] - f0))
        sparse_ti0 = t0.copy()
        sparse_ti0['ctime'][:] = t0['ctime'][:] + (t_00 - t0['ctime'][iifreq]) # ti0 if these were all the frequencies we cared about. easy!

        #...but, we always need all 1024 frequencies! need to interpolate reasonably.
        ti0 = extrapolate_to_full_band(sparse_ti0, im_freq['id'][:])
        frames = self.atime2frame(*ctimeo2atime(ti0['ctime'],ti0['ctime_offset']))
        if return_ctimeo:
            return ti0['ctime'],ti0['ctime_offset'],frames
        else:
            return frames
        
    def _ti0_from_t00_dm(self, t00, f0, dm, fi,return_ctimeo = False):
        """Get offsets per frequency

        Parameters
        ----------
        t00 : astropy.Time
            Of shape (1,)
        
        f0 : float64
            Of shape (1,)
        
        dm : np.array
            Of shape (npointing,)

        fi : np.array
            Of shape (nfreq = 1024,)

        return_ctimeo : bool
            If True, returns ctime, ctime_offset
            If False, returns integer frame numbers.
 
        Returns
        -------
        frames : np.array of shape (1024,n_pointing) and dtype = int
            Frame number of integration start.
        ctime, ctime_offset : np.arrays of shape (1024,n_pointing) and dtype = float64
            Returned if return_ctimeo is True.
            ctime_offset guaranteed to be smaller than float64 precision on the Unix time (~40 ns).
        """
        ti0 = t00 + K_DM * dm[None,:] * (fi[:,None]**-2 - f0**-2)*un.s  # for fi = infinity, ti0 < t00. Also, ti0 is an astropy.Time
        if return_ctimeo:
            ctime,ctime_offset = atime2ctimeo(ti0)
            return ctime, ctime_offset, self.atime2frame(ti0)
        return self.atime2frame(ti0)

    def _ti0_even_spacing(self,
        ti0,  
        period_frames, 
        num_scans_before = 0, 
        num_scans_after = 'max',
        time_ordered = False,
        return_ctimeo = True,
    ):
        """
        ti0 : np.array of float64 of shape (n_freq, n_pointing)
            Containing start times at station A, good to 2.56 us.
        period_frames : np.array of ints of shape (n_pointing).
            Spacing between successive scans in frames. 
            Could be an int or an array of ints -- one per pointing.
        num_scans_before : int or "max"
            Number of scans before and after.
        num_scans_after : int or "max"
            Number of scans before and after.
        Returns
        -------
        ctime : np.array of float64
            If return_ctimeo is True, will return ctime & ctime_offset
        
        ctime_offset : np.array of float64
            If return_ctimeo is True, will return ctime & ctime_offset
        
        tij : np.array of int
            Frame indices corresponding to ctime & ctime_offset with respect to self.bbdata_ref.
        """
        bbdata_ref = self.bbdatas[self.ref_index]
        time0 = bbdata_ref['time0'][:].copy()
        im_freq = bbdata_ref.index_map['freq']
        ntime = bbdata_ref.ntime
        assert np.issubdtype(ti0.dtype, np.integer), "Expected indices for ti0; got floats instead (perhaps Unix times?); probably wrong inputs."
        assert period_frames.shape == (ti0.shape[1],), "Maybe period_frames needs to be a 1d array of shape (n_pointing)"
        if num_scans_before == 'max': # attempt to use the whole dump
            num_scans_before = np.max(ti0 // period_frames) # shape = (1,)
        if num_scans_after == 'max': # attempt to use the whole dump
            num_scans_after = np.max((ntime - ti0) // period_frames) - 1 # minus one, since "num_scans_after" does not include the scan starting at ti0.

        scan_numbers = np.hstack((np.arange(0,num_scans_after + 1),
                                np.arange(-num_scans_before,0))
                                ) 
        print(scan_numbers)
        # this makes the scan number corresponding to ti0 come first in the array: e.g. if the on-pulse comes in the 5th scan period, the t_ij array is ordered as (5,6,7,...0,1,2,3,4)
        if time_ordered: # this makes the scans time-ordered, e.g. 0,1,2,3,...8,9.
            scan_numbers = sorted(scan_numbers) 
        tij = ti0[:, :, None] + scan_numbers[None, None, :] * period_frames[None, :, None] 
        # tij.shape = (n_freq, n_pointing, n_time); tij.dtype = int
        # Different period for different pointings
        # Different start time at different freq & pointing
        # Different scan number for each time

        if (tij < 0).any():
            logging.info(f'WARNING: Some ({100 * np.sum(tij < 0) / tij.size:.2f}%) of scans start before data dumped at reference station (likely CHIME). Did you specify your scan correctly?')
        if (tij > ntime).any():
            logging.info(f'WARNING: Some ({100 * np.sum(tij < 0) / tij.size:.2f}%) of scans start after data dumped at reference station (likely CHIME). Did you specify your scan correctly?')

        if return_ctimeo:
            ctime,ctime_offset =  atime2ctimeo(self.frame2atime(tij))
            return ctime, ctime_offset,tij
        else:
            return tij

    def visualize_twr_sea(self,
        bbdata_A,
        wfall,
        gate_start_frame,
        w,
        r,
        iiref=0,
        pointing = 0,
        dm = None,
        fscrunch = 4, 
        tscrunch = None,
        vmin=0,vmax=1,
        xpad=None,
        out_file:Optional[str]=None,
        bad_rfi_channels=None):
        wwfall = np.abs(wfall)**2
        wwfall -= np.nanmedian(wwfall,axis = -1)[:,:,None]
        wwfall /= median_abs_deviation(wwfall,axis = -1,nan_policy='omit')[:,:,None]
        if tscrunch is None:
            tscrunch = int((np.median(w) // 10 ))
        sww = _scrunch(wwfall,fscrunch = fscrunch, tscrunch = tscrunch)
        del wwfall

        y = np.arange(1024)
        for iiscan in range(t.shape[-1]):
            f = plt.figure()
            waterfall=sww[:,pointing] + sww[:,pointing+1]
            waterfall-=np.nanmedian(waterfall)
            plt.imshow(waterfall,aspect = 'auto',vmin = vmin,vmax = vmax,interpolation = 'none')

            x_start = gate_start_frame[:,pointing,iiscan]/ (tscrunch)
            
            x_end = x_start + w[pointing,iiscan] / tscrunch
            x_mid = x_start + (x_end - x_start) * 0.5 
            x_rminus = x_mid - (x_end - x_start) * 0.5 * r[:,pointing,iiscan]
            x_rplus = x_mid + (x_end - x_start) * 0.5 * r[:,pointing,iiscan]
            plt.fill_betweenx(x1 = x_start, x2 = x_end,y = y/fscrunch,alpha = 0.15)
            if iiscan == 0:
                linestyle = '-'
            else:
                linestyle = '--'
            plt.plot(x_start, y/fscrunch,linestyle = linestyle,color = 'black',label='window',lw=1) # shade t
            plt.plot(x_end, y/fscrunch,linestyle = linestyle,color = 'black',lw=1) # shade t + w
            if bad_rfi_channels is not None:
                for channel in bad_rfi_channels:
                    plt.axhline(channel/fscrunch,color='gray',alpha=.25)
            plt.plot(x_rminus, y/fscrunch,linestyle = '-.',color = 'red',label='integration',lw=1) # shade t + w/2 - r/2
            plt.plot(x_rplus, y/fscrunch,linestyle = '-.',color = 'red',lw=1) # shade t + w/2 + r/2
            plt.legend(loc='lower right')
            if t_a_type=='unix':
                xmin = np.nanmin((gate_start_frame[:,pointing,:] - bbdata_A['time0']['ctime'][:,None]).sec,axis = -1) / (2.56e-6 * tscrunch)
                xmax = np.nanmax((gate_start_frame[:,pointing,:] - bbdata_A['time0']['ctime'][:,None]).sec,axis = -1) / (2.56e-6 * tscrunch)
            else:
                xmin = np.nanmin(gate_start_frame[:,pointing,:],axis = -1) / (tscrunch)
                xmax = np.nanmax(gate_start_frame[:,pointing,:],axis = -1) / (tscrunch)
            if xpad is not None:
                plt.xlim(np.nanmedian(xmin)-xpad,np.nanmedian(xmax)+xpad)
            plt.ylim(1024 / fscrunch,0)
            plt.ylabel(f'Freq ID (0-1023) / {fscrunch:0.0f}')
            plt.xlabel(f'Time ({tscrunch:0.1f} frames)')
            if out_file is not None:
                fig.savefig(out_file,bbox_inches='tight')
        del bbdata_A
        return f

    def tij_other_stations(self, gate_spec, pointing_spec):
        """Do this on a per-pointing and per-station basis.
        
        Parameters
        ----------
        gate_spec : np.ndarray of dtype VLBIVis._dataset_dtype['time'] and shape (n_freq, n_pointing, n_time)
            Holds attributes 'corr_ra', 'corr_dec', 'source_name', 'dm_correlator'.
        
        pointing_spec : np.ndarray of type VLBIVis._dataset_dtype['pointing'] of shape (n_pointing).
        """
        iiref = self.tel_names.index(self.ref_station)
        tij_unix = gate_spec['gate_start_unix'] #tij_unix.shape = (n_freq, n_pointing, n_time)
        n_tel = len(self.telescopes)
        n_freq = 1024
        n_pointing = len(pointing_spec)
        n_time = gate_spec.shape[-1]

        tij_sp = np.zeros(
            (n_tel,n_freq,n_pointing,n_time),
            dtype = int) 
        # Double check pointings are OK
        assert np.isclose(self.pycalc_results.src_ra.deg,pointing_spec['corr_ra']).all(), "pycalc does not match inputted pointing_spec"
        assert np.isclose(self.pycalc_results.src_dec.deg,pointing_spec['corr_dec']).all(), "pycalc does not match inputted pointing_spec"
        for ii in range(len(pointing_spec)):
            delays= self.pycalc_results.interpolate_delays(
                Time(tij_unix[:,ii,:].flatten(),format = 'unix'))[:,0,:,:] 
                #delays.shape = (n_freq * n_time,n_station, n_pointing) 
            delays -= delays[:, iiref, None,:] # subtract delay at the reference station -- now we have instantaneous baseline delays of shape (n_freq * n_time, n_station, n_pointing)
        for iitel, telescope in enumerate(self.telescopes):
            for jjpointing, pointing in enumerate(self.pointings):
                tau_ij = delays[:, iitel, jjpointing].reshape(tij[:,jjpointing,:].shape)/2.56 #frames
                tij_sp[iitel,:,jjpointing,:] = tij[:,jjpointing,:] + tau_ij
        self.tij_sp = tij_sp
        return tij_sp
    
    def run_correlator_job( 
            self,
            gate_spec,
            pointing_spec = None,
            max_lag = 100,
            out_h5_file = None,
            auto_corr:bool=False,
        ):
        """Run auto- and cross- correlations.

        Loops over baselines, then frequencies, then pointings.
        Are all read in at once and ordered using fill_waterfall. This works well on short baseband dumps. 
        Memory cost: 2 x BBData, 
        I/O cost:N*(N-1) / 2 x BBData.

        Parameters
        ----------
        gate_start_frame : np.ndarray
            Of topocentric start indices for the on-signal gating relative to the start of the dump as a function of (n_station, n_freq, n_pointing, n_time)
        w_ij : np.ndarray
            Of start times as a function of (n_pointing, n_time)
        r_ij : np.ndarray
            Of start times as a function of (n_freq, n_pointing, n_time)
        ref_index : intdec_target
            index corresponding to bbdata where topocentric time is defined (CHIME)
        dm : float
            A dispersion measure for de-smearing. Fractional precision needs to be 10%.
        auto_corr: bool
            If True, also calculate autocorrelations
        """
        if pointing_spec is None:
            pointing_spec = self.pointing_spec
            
        t_ij_station_pointing = self.tij_other_stations(gate_spec=gate_spec,pointing_spec=pointing_spec)

        gate_start_frame = gate['gate_start_frame'] 
        w_ij = gate['window'] 
        r_ij = gate['r'] 
        dm = gate['dm'] 
        ref_index=self.tel_names.index(self.ref_station)
        ref_index=self.ref_index
        bbdata_ref = self.bbdatas[self.ref_index]
        tel_bbdatas=self.bbdatas

        gate_start_frame_top=gate_start_frame[self.ref_index]


        if auto_corr:
            for iia, bbdata_a in enumerate(tel_bbdatas):

                assert np.issubdtype(gate_start_frame.dtype, np.integer), "gate_start_frame must be an integer start frame"

                gate_start_frame_tel = gate_start_frame[iia] #extract start frame for station

                # there are scans with missing data: check the start and end index
                mask_a = (gate_start_frame_tel < 0) + (gate_start_frame_tel  + w_ij[None,:,:] > bbdata_a.ntime) 

                gate_start_frame_tel[mask_a] = int(bbdata_a.ntime // 2)
                # ...but we just let the correlator correlate
                logging.info(f'Calculating autos for station {iia}')

                auto_vis = autocorr_core(DM=dm, bbdata_a=bbdata_a, 
                                        t_a = gate_start_frame_tel,
                                        window = w_ij,
                                        R = r_ij,
                                        max_lag =self.max_lag, 
                                        n_pol = 2)

                # ...and replace with nans afterward.
                auto_vis += mask_a[:,:,None,None,None,:] * np.nan # fill with nans where
                gate_start_unix=bbdata_a['time0']['ctime'][:,np.newaxis,np.newaxis]*np.ones(gate_start_frame_tel.shape)
                gate_start_unix_offset=bbdata_a['time0']['ctime_offset'][:,np.newaxis,np.newaxis]+gate_start_frame_tel*2.56e-6
                output._from_ndarray_station(
                    event_id,
                    telescope = self.telescopes[iia],
                    bbdata = bbdata_a,
                    auto = auto_vis,
                    gate_start_frame=gate_start_frame_tel,
                    gate_start_unix=gate_start_unix,
                    gate_start_unix_offset=gate_start_unix_offset,
                    window=w_ij,
                    r=r_ij,
                    )
                logging.info(f'Wrote autos for station {iia}')
        
        cross = cross_correlate_baselines(
                bbdatas=tel_bbdatas,
                bbdata_top=bbdata_ref,
                t_a=gate_start_frame_top,
                window=w_ij,
                R=r_ij,
                pycalc_results=self.pycalc_results,
                DM=dm,
                station_indices=np.array(range(len(tel_bbdatas))),
                max_lag=self.max_lag, 
                n_pol=2,
                weight=None,
                ref_frame=ref_index,
                fast=True
            )
        m=0
        for telA in range(len(tel_bbdatas)-1):
            gate_start_frame_top = gate_start_frame[telA] #extract start frame for station
            for telB in range(telA+1,len(tel_bbdatas)):
                output._from_ndarray_baseline(
                    event_id = event_id,
                    pointing_center=pointing_centers,
                    telescope_a=self.telescopes[telA],
                    telescope_b=self.telescopes[telB],
                    cross=cross[m], 
                    gate_start_unix=bbdata_ref["time0"]["ctime"][:,np.newaxis,np.newaxis]*np.ones(gate_start_frame_top.shape),
                    gate_start_unix_offset=bbdata_ref["time0"]["ctime_offset"][:,np.newaxis,np.newaxis]+gate_start_frame_top*2.56e-6,
                    window=w_ij,
                    r=r_ij
                )
                m+=1
                
                logging.info(f'Wrote visibilities for baseline {telA}-{telB}')
        del tel_bbdatas # free up space in memory

        if type(out_h5_file) is str:
            output.save(out_h5_file)
            logging.info(f'Wrote visibilities to disk: ls -l {out_h5_file}')
        return output