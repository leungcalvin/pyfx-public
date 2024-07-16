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

import datetime
import logging
import os
from decimal import Decimal
from typing import List, Optional, Tuple

import astropy
import astropy.coordinates as ac
import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import _scrunch, fill_waterfall
from coda.core import VLBIVis
from pycalc11 import Calc
from scipy.fft import fft, ifft, next_fast_len
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation

from pyfx.bbdata_io import get_all_im_freq, get_all_time0, station_from_bbdata
from pyfx.core_correlation_station import autocorr_core, crosscorr_core,fringestop_station

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


def duration_frames(arr, axis=-1):
    """Gets number of valid samples counted over the specified axis."""
    return np.sum(np.isfinite(arr), axis=axis)


def tw2twr_frames(t: np.ndarray, w):
    """Converts from freq dependent w to freq dependent r.
    Here t is assumed to be an index; therefore it is quantized to integers.
    Up to rounding errors in w, this is the inverse of tw2twr()."""
    r_new = w / np.max(w)  # one per frequency
    w_new = np.max(w)  # one per band
    return t.astype(int), w_new.astype(int), r_new


def twr2tw_frames(t, w, r):
    """Converts from freq dependent r to freq dependent w.
    Here t is assumed to be an index; therefore it is quantized to intege
    Up to rounding errors in w, this is the inverse of tw2twr().
    """
    return t.astype(int), (w * r).astype(int)


K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
FREQ = np.linspace(800, 400, num=1024, endpoint=False)


def right_broadcasting(arr, target_shape):
    return arr.reshape(arr.shape + (1,) * (len(target_shape) - arr.ndim))


def same_pointing(bbdata_a, bbdata_b):
    assert np.isclose(
        bbdata_a["tiedbeam_locations"]["ra"][:], bbdata_b["tiedbeam_locations"]["ra"][:]
    ).all()
    assert np.isclose(
        bbdata_a["tiedbeam_locations"]["dec"][:],
        bbdata_b["tiedbeam_locations"]["dec"][:],
    ).all()
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
    assert (
        np.max(w_ij * 2.56e-6) < earth_rotation_time
    ), "Use smaller value of w_ij, scans are too long!"

    if dm is not None:  # check that wij exceeds smearing time for that pointing
        dm_smear_sec = K_DM * dm[None, :, None] * 0.390625 / freq[:, None, None] ** 3
        diff = np.min(
            w_ij * 2.56e-6 - dm_smear_sec, axis=(-2, -1)
        )  # check all frequency channels
        if not (diff > 0).all():
            logging.warning(
                f"For DM = {dm}, w_ij needs to be increased by {-(np.min(diff)/ 2.56e-6):0.2f} frames to not clip the pulse within a channel"
            )
    return w_ij


def extrapolate_to_full_band(time0: np.ndarray, freq_ids: np.ndarray):
    """Interpolates time0 to the full bandwidth to float64 precision, which is sufficient for specifying start times.

    Does this by:
    1) calculating the fpga_start_time with high precision,
    2) doing nearest-neighbor interp on fpga_start_time,
    3) then doing high-precision calculation of ctime and ctime_offset
    4) interpolating at the frequencies we do not have.

    This does something very similar to baseband_analysis.fill_waterfall()!

    """
    new_t0 = np.zeros((1024,), dtype=time0.dtype)
    new_fpga_start_time_unix = interp1d(
        x=freq_ids,
        y=fpga_start_time(time0, start_time_error=True, astropy_time=False).astype(
            float
        ),
        kind="nearest",
        fill_value=(time0["fpga_count"][0], time0["fpga_count"][-1]),
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
    new_t0["ctime"][freq_ids] = time0["ctime"]
    new_t0["ctime_offset"][freq_ids] = time0["ctime_offset"]
    new_t0["fpga_count"][freq_ids] = time0["fpga_count"]
    return new_t0


def initialize_pointing_from_coords(ras, decs, names, dm_trials):
    assert ras.shape == decs.shape
    assert names.shape == dm_trials.shape
    assert ras.size == names.size * 2
    pointing_spec = np.empty(names.shape, dtype=VLBIVis._dataset_dtypes["pointing"])
    pointing_spec["corr_ra"][:] = ras[::2]
    pointing_spec["corr_dec"][:] = decs[::2]
    pointing_spec["source_name"][:] = names[:]
    pointing_spec["corr_dm"][:] = dm_trials[:]
    return pointing_spec


def initialize_pointing_dm_trials(single_ra, single_dec, single_name, dm_trials):
    assert ras.shape == decs.shape
    assert names.shape == dm_trials.shape
    assert ras.size == names.size * 2
    pointing_spec = np.empty(names.shape, dtype=VLBIVis._dataset_dtypes["pointing"])
    pointing_spec["corr_ra"][:] = single_ra
    pointing_spec["corr_dec"][:] = single_dec
    pointing_spec["source_name"][:] = np.array(
        [single_name + f"_DMT_{ii:0>3d}" for ii in range(len(dm_trials))]
    )
    pointing_spec["corr_dm"][:] = dm_trials[:]
    return pointing_spec


def atime2ctimeo(at):
    """Convert astropy.Time into ctime & ctime_offset"""
    ctime_long = at.to_value("unix", "long")  # convert to float128
    ctime = ctime_long.astype(np.float64)  # break into float64
    ctime_offset = (ctime_long - ctime).astype(
        np.float64
    )  # take the remainder as another float64
    return ctime, ctime_offset


def ctimeo2atime(ctime, ctime_offset):
    """Convert ctime & ctime_offset into astropy.Time"""
    return Time(val=ctime, val2=ctime_offset, format="unix")


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


class CorrJob:
    def __init__(
        self,
        pointing_spec: np.ndarray,
        bbdatas: List[BBData],
        telescopes: List[ac.earth.EarthLocation],
        bbdata_filepaths: Optional[List[str]] = None,
        ref_station: Optional[str] = "chime",
        default_max_lag=100,
    ):
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
        self.pointing_spec = pointing_spec

        # get tel names
        for i, this_bbdata in enumerate(bbdatas):
            tel_name = station_from_bbdata(
                this_bbdata,
            )
            if tel_name != self.tel_names[i]:
                print(
                    f"warning: telescope name {self.tel_names[i]} from input telescopes does not match telescope name {tel_name} from bbdata. Please check that your input parameters are identically ordered."
                )
        self.telescopes = telescopes
        self.ref_station = ref_station
        ref_index = self.tel_names.index(ref_station)
        self.ref_index = ref_index
        bbdata_ref = self.bbdatas[self.ref_index]

        earliest_start_unix = np.inf
        latest_end_unix = -np.inf

        # loop again to check data format specifications
        for i, this_bbdata in enumerate(bbdatas):
            assert same_pointing(bbdata_ref, this_bbdata)
            earliest_start_unix = min(
                earliest_start_unix, this_bbdata["time0"]["ctime"][0]
            )
            latest_end_unix = max(
                latest_end_unix, this_bbdata["time0"]["ctime"][-1] + this_bbdata.ntime
            )
            if i == ref_index:
                self.ref_ctime = this_bbdata["time0"]["ctime"]
                self.ref_ctime_offset = this_bbdata["time0"]["ctime_offset"]
            if this_bbdata.nfreq < 1024:
                fill_waterfall(this_bbdata, write=True)

        earliest_start_unix = int(earliest_start_unix - 1)  # buffer
        duration_min = 3  # max(int(np.ceil(int(latest_end_unix - earliest_start_unix + 1.0 )/60)),1)
        self.pointings_sc = ac.SkyCoord(
            ra=self.pointing_spec["corr_ra"].flatten() * un.deg,
            dec=self.pointing_spec["corr_dec"].flatten() * un.deg,
            frame="icrs",
        )
        duration_min = 1
        ci = Calc(
            station_names=self.tel_names,
            station_coords=self.telescopes,
            source_coords=self.pointings_sc,
            start_time=Time(np.floor(earliest_start_unix), format="unix", precision=9),
            duration_min=duration_min,
            base_mode="geocenter",
            dry_atm=True,
            wet_atm=True,
            d_interval=1,
        )
        ci.run_driver()
        self.pycalc_results = ci
        self.max_lag = default_max_lag  # default value if define_scan_params is not called (continuum sources)
        return

    def t0_f0_from_bbdata(self, t0f0: Tuple[str], bbdata_ref, return_ctimeo=False):
        """Returns the actual t00 and f0 from bbdata_ref, which could be either a BBData object or a filepath.

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
            im_freq = bbdata_ref.index_map["freq"]
            time0 = bbdata_ref["time0"]
            ntime = bbdata_ref.ntime
        elif type(bbdata_ref) is str:
            im_freq = get_all_im_freq(bbdata_ref)
            time0 = get_all_time0(bbdata_ref)
            ntime = get_ntime(bbdata_ref)

        (_t0, _f0) = t0f0
        if _f0 == "top":  # use top of the collected band as reference freq
            iifreq = 0
        elif _f0 == "bottom":  # use bottom of the collected band as reference freq
            iifreq = -1
        elif isinstance(_f0, float): #numpy64 will fail on this -> type(_f0) is float:  # use the number as the reference freq
            iifreq = np.argmin(np.abs(im_freq["centre"][:] - _f0))
            offset_mhz = im_freq["centre"][iifreq] - _f0
            logging.info(
                "INFO: Offset between requested frequency and closest frequency: {offset_mhz} MHz"
            )
        else:
            raise AssertionError("Please pass in t0f0: (astropy.Time,float) or t0f0: (astropy.Time,str) where f0='top' or 'bottom")
        f0 = im_freq["centre"][iifreq]  # the actual reference frequency in MHz.

        if _t0 == "start":
            t0 = Time(
                time0["ctime"][iifreq],
                val2=time0["ctime_offset"][iifreq],
                format="unix",
            )  # the actual reference start time
        if _t0 == "middle":
            t0 = Time(
                time0["ctime"][iifreq] + ntime * 2.56e-6 // 2,
                val2=time0["ctime_offset"][iifreq],
            )
        else:
            t0 = _t0
        return t0, f0

    def define_scan_params_continuum(self, equal_duration=True, pad=100):
        """Return start frame and number of window frames ("t" and "w") by looking at the nan pattern in telA_bbdata, making use of ~all the data we have.

        This is an appropriate way to get the t,w,r data for a single phase-center pointing on a continuum source.

        To take into account that the different boundaries of the data, we trim the edges of the scan.
        We remove :pad: frames from both the left and right of the integration.gate_start_frame & ww.

        Parameters
        ----------
        self : CorrJob
            Should have self.bbdatas and self.ref_index attributes.

        equal_duration : bool
            If equal_duration is set to True, we will make sure the integration time is the same across all frequencies.

        Returns
        -------
        gate_spec : np.array of shape (1024,n_pointings,1)
            Note that there is only one scan because we're integrating over the whole dump.
        """
        bbdata_ref = self.bbdatas[self.ref_index]
        pol = 0
        n_channels = 1024
        wfall_data = bbdata_ref["tiedbeam_baseband"][
            :, pol, :
        ]  # both polarizations should have same nan pattern because Kotekan works in freq  time element order.
        assert (
            wfall_data.shape[0] == n_channels
        ), f"Frequency channels missing, please pass in {n_channels} channels"
        gate_start_frame = first_valid_frame(wfall_data, axis=-1)
        window = duration_frames(wfall_data, axis=-1)
        if wfall_data.shape[-1] // 2 > np.median(window):
            logging.info(
                f'WARNING: More than half of the channels invalid in data for {telA_bbdata.attrs["event_id"]}'
            )
        if (
            equal_duration
        ):  # choose the length that maximizes the sensitivity as quantified by num_valid_channels x duration
            sorted_window_lengths = np.sort(window)
            sens_metric = sorted_window_lengths * (
                1024 - np.arange(1024)
            )  # n_time * n_channels with that many valid samples
            window = sorted_window_lengths[np.argmax(sens_metric)] + np.zeros(
                1024, dtype=int
            )
        assert (
            np.min(window) - 2 * pad > 0
        ), "twr params result in negative integration duration when zero-padded. Please optimize manually, e.g. decrease pad value or input twr manually."
        tt, ww, rr = tw2twr_frames(gate_start_frame + pad, window - 2 * pad)
        ww = np.atleast_2d(ww)
        tt.shape = (1024, len(self.pointing_spec), 1)
        rr.shape = (1024, len(self.pointing_spec), 1)

        gate_spec = np.empty(tt.shape, dtype=VLBIVis._dataset_dtypes["time"])
        gate_spec["gate_start_unix"], gate_spec["gate_start_unix_offset"] = (
            atime2ctimeo(self.frame2atime(tt))
        )
        gate_spec["gate_start_frame"] = tt
        gate_spec["duration_frames"] = ww
        gate_spec["dur_ratio"] = rr
        return gate_spec

    def define_scan_params_transient(
        self,
        t0f0: Optional[Tuple[float, float]] = None,
        start_or_toa="start",
        freq_offset_mode="bbdata",
        time_spacing="even",
        window=1000,
        r_ij=np.ones(1024),
        num_scans_before=10,
        num_scans_after=8,
        time_ordered=False,
        period_frames=1000,
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

        Window : np.ndarray, of shape (npointing,)
            Sets the integration duration in frames as a function of pointing.

        period_frames : np.ndarray, of shape (npointing,)
            Sets the spacing between integrations as a function of pointing.
        kwargs : 'dm' and 'f0', 'pdot','wi'
        """
        period_frames = np.atleast_1d(period_frames)
        dm = self.pointing_spec["dm_correlator"]
        bbdata_ref = self.bbdatas[self.ref_index]

        # First: if t0f0 is a bunch of strings, then get t0 & f0 from the BBData.
        # t00 will be output as an astropy.Time
        # f0 will be output as a float
        t00, f0 = self.t0_f0_from_bbdata(t0f0, bbdata_ref=bbdata_ref)

        # First do t_i0 for the reference station, i.e. generate start times for other frequencies.
        if freq_offset_mode == "bbdata":
            t_i0 = self._ti0_from_t00_bbdata(
                bbdata_ref, t_00=t00, f0=f0, return_ctimeo=False
            )
        elif freq_offset_mode == "dm":
            assert (
                type(t0f0[0]) is not str and type(t0f0[1]) is not str
            ), "You probably want to pass in a hard-coded time & frequency reference as t0f0 if you want to follow a DM sweep."
            t_i0 = self._ti0_from_t00_dm(
                t00, f0, dm=dm, fi=FREQ, return_ctimeo=False
            )  # frame indices
        # If _ti0 is a TOA, need to shift _ti0 back by half a scan length
        else:
            raise ValueError('freq_offset_mode must be either "bbdata" or "dm"')

        _tij = t_i0 # t_i0.shape = (n_freq, n_pointing)
         # by default, only do one scan
        # Next do t_ij for the reference station from t_i0.

        if num_scans_before or num_scans_after:
            period_frames = np.broadcast_to(period_frames,shape = dm.shape) # broadcast to (n_pointing,)
            if time_spacing == "even":
                _tij = self._ti0_even_spacing(
                    t_i0,
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
            if time_spacing == "overlap2":
                _tij1 = self._ti0_even_spacing(
                    t_i0,
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
                _tij2 = self._ti0_even_spacing(
                    t_i0 + int(0.5 * period_frames),
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
                _tij = np.concatenate(
                    (_tij1, _tij2), axis=-1
                )  # concatenate along time axis
            if time_spacing == "overlap3":
                _tij1 = self._ti0_even_spacing(
                    t_i0,
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
                _tij2 = self._ti0_even_spacing(
                    t_i0 + int(period_frames / 3),
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
                _tij3 = self._ti0_even_spacing(
                    t_i0 + int(2 * period_frames / 3),
                    period_frames,
                    num_scans_before=num_scans_before,
                    num_scans_after=num_scans_after,
                    time_ordered=time_ordered,
                    return_ctimeo=False,
                )
                _tij = np.concatenate(
                    (_tij1, _tij2, _tij3), axis=-1
                )  # concatenate along time axis

        window = np.atleast_2d(window)
        window = right_broadcasting(window, target_shape=_tij.shape)
        r_ij = np.atleast_2d(r_ij)
        r_ij = right_broadcasting(r_ij, target_shape=_tij.shape)
        assert np.issubdtype(
            window.dtype, np.integer
        ), "Window must be an integer number of frames!"
        if start_or_toa == "toa":
            _tij -= window  # Scan duration given by :Window:, not :period_frames:!
            logging.info(
                "INFO: Using TOA mode: shifting all scans to be centered on t_ij"
            )

        logging.info(
            "Success: generated a valid set of integrations! Rounding to nearest 2.56us to create topocentric gate specification."
        )
        t_ij = self.frame2atime(int_frames=_tij)
        window = np.broadcast_to(
            right_broadcasting(window, target_shape=t_ij.shape), t_ij.shape
        )
        gate_spec = np.empty(_tij.shape, dtype=VLBIVis._dataset_dtypes["time"])
        gate_spec["gate_start_unix"], gate_spec["gate_start_unix_offset"] = (
            atime2ctimeo(t_ij)
        )
        gate_spec["gate_start_frame"] = _tij
        gate_spec["duration_frames"] = window + np.zeros_like(t_ij)
        gate_spec["dur_ratio"] = r_ij
        validate_wij(
            gate_spec["duration_frames"], r_ij, dm=dm
        )  # check DM smearing not too large
        return gate_spec

    def atime2frame(self, timestamps: np.ndarray, bbdata=None, force=False):
        """Convert astropy.Time to frame index.
        For safety, if a bbdata besides self.ref_bbdata is desired,
        you must pass force = True must be passed in. Or else we disallow it.

        timestamps : astropy.Time whose shape is (1024,...)

        bbdata : BBData
            If passed in and force = True, will grab reference times from this BBData instead.

        force : bool
            For safety, set to False by default.
        """
        if force and bbdata is not None:
            ctime = bbdata["time0"]["ctime"][:].copy()
            ctime_offset = bbdata["time0"]["ctime_offset"][:].copy()
        elif bbdata is not None and not force:
            raise ValueError(
                "Are you sure you want to calculate a frame index for a non-core station? If so, pass force = True"
            )
        else:
            ctime = self.ref_ctime
            ctime_offset = self.ref_ctime_offset
        _ctime = Time(
            ctime,
            val2=ctime_offset,
            format="unix",
            precision=9,
        )
        _ctime_shaped = right_broadcasting(_ctime, target_shape=timestamps.shape)
        closest_frame = np.round(
            ((timestamps - _ctime_shaped).sec / (2.56e-6 * un.s)).value
        ).astype(int)
        return closest_frame  # timestamps_rounded

    def frame2atime(self, int_frames: np.ndarray, bbdata=None, force=False):
        """Convert frame index to astropy.Time"""
        if force and bbdata is not None:
            ctime = bbdata["time0"]["ctime"][:].copy()
            ctime_offset = bbdata["time0"]["ctime_offset"][:].copy()
        elif bbdata is not None and not force:
            raise ValueError(
                "Are you sure you want to calculate a frame index for a non-core station? If so, pass force = True"
            )
        else:
            ctime = self.ref_ctime
            ctime_offset = self.ref_ctime_offset
        exact_atime = Time(
            right_broadcasting(ctime, int_frames.shape) + 2.56e-6 * int_frames,
            val2=right_broadcasting(ctime_offset, int_frames.shape),
            format="unix",
            precision=9,
        )
        return exact_atime

    def _ti0_from_t00_bbdata(self, bbdata, t_00, f0, return_ctimeo=False):
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
        ti0 : np.array of shape (1024, n_pointing) of true times.
        """
        if type(bbdata) is str:
            im_freq = get_all_im_freq(bbdata_filename)
            t0 = get_all_time0(bbdata_filename)
        elif type(bbdata) is BBData:
            im_freq = bbdata.index_map["freq"]
            t0 = bbdata["time0"][:].copy()
        iifreq = np.argmin(np.abs(im_freq["centre"][:] - f0))

        if t_00 == "start":
            t_00 = t0["ctime"][iifreq]
        sparse_ti0 = t0.copy()
        sparse_ti0["ctime"][:] = t0["ctime"][:] + (t_00 - t0["ctime"][iifreq])
        # ti0 if these were all the frequencies we cared about. easy!

        # ...but, we always need all 1024 frequencies! need to interpolate reasonably.
        ti0 = extrapolate_to_full_band(sparse_ti0, im_freq["id"][:])
        ti0 = right_broadcasting(ti0,(1024,len(self.pointing_spec)))
        frames = self.atime2frame(ctimeo2atime(ti0["ctime"], ti0["ctime_offset"]))
        if return_ctimeo:
            return ti0["ctime"], ti0["ctime_offset"], frames
        else:
            return frames

    def _ti0_from_t00_dm(self, t00, f0, dm, fi, return_ctimeo=False):
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
        ti0 = (
            t00 + K_DM * dm[None, :] * (fi[:, None] ** -2 - f0**-2) * un.s
        )  # for fi = infinity, ti0 < t00. Also, ti0 is an astropy.Time
        if return_ctimeo:
            ctime, ctime_offset = atime2ctimeo(ti0)
            return ctime, ctime_offset, self.atime2frame(ti0)
        return self.atime2frame(ti0)

    def _ti0_even_spacing(
        self,
        ti0,
        period_frames,
        num_scans_before=0,
        num_scans_after="max",
        time_ordered=False,
        return_ctimeo=True,
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
        time0 = bbdata_ref["time0"][:].copy()
        im_freq = bbdata_ref.index_map["freq"]
        ntime = bbdata_ref.ntime
        assert np.issubdtype(
            ti0.dtype, np.integer
        ), "Expected indices for ti0; got floats instead (perhaps Unix times?); probably wrong inputs."
        assert period_frames.shape == (
            ti0.shape[1],
        ), "Maybe period_frames needs to be a 1d array of shape (n_pointing)"
        if num_scans_before == "max":  # attempt to use the whole dump
            num_scans_before = np.max(ti0 // period_frames)  # shape = (1,)
        if num_scans_after == "max":  # attempt to use the whole dump
            num_scans_after = (
                np.max((ntime - ti0) // period_frames) - 1
            )  # minus one, since "num_scans_after" does not include the scan starting at ti0.

        scan_numbers = np.hstack(
            (np.arange(0, num_scans_after + 1), np.arange(-num_scans_before, 0))
        )
        # this makes the scan number corresponding to ti0 come first in the array: e.g. if the on-pulse comes in the 5th scan period, the t_ij array is ordered as (5,6,7,...0,1,2,3,4)
        if time_ordered:  # this makes the scans time-ordered, e.g. 0,1,2,3,...8,9.
            scan_numbers = sorted(scan_numbers)
        tij = (
            ti0[:, :, None] + scan_numbers[None, None, :] * period_frames[None, :, None]
        )
        # tij.shape = (n_freq, n_pointing, n_time); tij.dtype = int
        # Different period for different pointings
        # Different start time at different freq & pointing
        # Different scan number for each time

        if (tij < 0).any():
            logging.info(
                f"WARNING: Some ({100 * np.sum(tij < 0) / tij.size:.2f}%) of scans start before data dumped at reference station (likely CHIME). Did you specify your scan correctly?"
            )
        if (tij > ntime).any():
            logging.info(
                f"WARNING: Some ({100 * np.sum(tij < 0) / tij.size:.2f}%) of scans start after data dumped at reference station (likely CHIME). Did you specify your scan correctly?"
            )

        if return_ctimeo:
            ctime, ctime_offset = atime2ctimeo(self.frame2atime(tij))
            return ctime, ctime_offset, tij
        else:
            return tij

    def visualize_twr_sea(
        self,
        bbdata_A,
        wfall,
        gate_spec,
        iiref=0,
        pointing=0,
        fscrunch=4,
        tscrunch=None,
        vmin=0,
        vmax=1,
        xpad=None,
        out_file: Optional[str] = None,
        bad_rfi_channels=None,
    ):
        gate_start_frame = gate_spec['gate_start_frame']
        w = gate_spec['duration_frames']
        r = gate_spec['dur_ratio']
        wwfall = np.abs(wfall) ** 2
        wwfall -= np.nanmedian(wwfall, axis=-1)[:, :, None]
        wwfall /= median_abs_deviation(wwfall, axis=-1, nan_policy="omit")[:, :, None]
        if tscrunch is None:
            tscrunch = int((np.median(w) // 10))
        sww = _scrunch(wwfall, fscrunch=fscrunch, tscrunch=tscrunch)
        del wwfall

        y = np.arange(1024)
        f = plt.figure()
        for iiscan in range(gate_start_frame.shape[-1]):
            waterfall = sww[:, pointing] + sww[:, pointing + 1]
            waterfall -= np.nanmedian(waterfall)
            plt.imshow(
                waterfall, aspect="auto", vmin=vmin, vmax=vmax, interpolation="none"
            )

            x_start = gate_start_frame[:, pointing, iiscan] / (tscrunch)

            x_end = x_start + w[:,pointing, iiscan] / tscrunch
            x_mid = x_start + (x_end - x_start) * 0.5
            x_rminus = x_mid - (x_end - x_start) * 0.5 * r[:, pointing, iiscan]
            x_rplus = x_mid + (x_end - x_start) * 0.5 * r[:, pointing, iiscan]
            plt.fill_betweenx(x1=x_start, x2=x_end, y=y / fscrunch, alpha=0.15)
            if iiscan == 0:
                linestyle = "-"
            else:
                linestyle = "--"
            plt.plot(
                x_start,
                y / fscrunch,
                linestyle=linestyle,
                color="black",
                label="window",
                lw=1,
            )  # shade t
            plt.plot(
                x_end, y / fscrunch, linestyle=linestyle, color="black", lw=1
            )  # shade t + w
            if bad_rfi_channels is not None:
                for channel in bad_rfi_channels:
                    plt.axhline(channel / fscrunch, color="gray", alpha=0.25)
            plt.plot(
                x_rminus,
                y / fscrunch,
                linestyle="-.",
                color="red",
                label="integration",
                lw=1,
            )  # shade t + w/2 - r/2
            plt.plot(
                x_rplus, y / fscrunch, linestyle="-.", color="red", lw=1
            )  # shade t + w/2 + r/2
            plt.legend(loc="lower right")

            xmin = np.nanmin(gate_start_frame[:, pointing, :], axis=-1) / (tscrunch)
            xmax = np.nanmax(gate_start_frame[:, pointing, :], axis=-1) / (tscrunch)
            if xpad is not None:
                plt.xlim(np.nanmedian(xmin) - xpad, np.nanmedian(xmax) + xpad)
            plt.ylim(1024 / fscrunch, 0)
            plt.ylabel(f"Freq ID (0-1023) / {fscrunch:0.0f}")
            plt.xlabel(f"Time ({tscrunch:0.1f} frames)")
            if out_file is not None:
                plt.savefig(out_file, bbox_inches="tight")
        del bbdata_A
        return f

    def tij_other_stations(self, gate_spec):
        """Do this on a per-pointing and per-station basis.

        For each gate and each pointing, figure out when the other stations start integrating in unix time.

        Parameters
        ----------
        gate_spec : np.ndarray of dtype VLBIVis._dataset_dtype['time'] and shape (n_freq, n_pointing, n_time)
            Holds attributes 'corr_ra', 'corr_dec', 'source_name', 'dm_correlator'.

        Returns
        -------
        tij_ctime_sp : np.ndarray of float64 of shape (n_tel, n_freq, n_pointing, n_time)
            Unix start times of the scan at each station
        tij_ctime_offset_sp : np.ndarray of float64 of shape (n_tel, n_freq, n_pointing, n_time)
            Unix start times of the scan at each station; guaranteed to be a small correction (< FLOAT64_PRECISION)
        tij_frame_sp : np.ndarray of int of shape (n_tel, n_freq, n_pointing, n_time)
            Unix start frame of the scan at each station.
            N.b. this is kept around for metadata but useless for now.
        """
        pointing_spec = self.pointing_spec
        iiref = self.tel_names.index(self.ref_station)
        tij_unix = gate_spec[
            "gate_start_unix"
        ]  # tij_unix.shape = (n_freq, n_pointing, n_time)
        n_tel = len(self.telescopes)
        n_freq = 1024
        n_pointing = len(self.pointing_spec)
        n_time = gate_spec.shape[-1]

        tij_unix_sp_coarse = np.zeros(
            (n_tel, n_freq, n_pointing, n_time), dtype=float
        )  # coarse times
        tij_ctime_sp = np.zeros_like(tij_unix_sp_coarse)  # fine times
        tij_ctimeo_sp = np.zeros_like(tij_unix_sp_coarse)  # fine times
        # Double check pointings are OK
        assert np.isclose(
            self.pycalc_results.src_ra.deg, self.pointing_spec["corr_ra"]
        ).all(), "pycalc does not match self.pointing_spec. initialize CorrJob again"
        assert np.isclose(
            self.pycalc_results.src_dec.deg, self.pointing_spec["corr_dec"]
        ).all(), "pycalc does not match self.pointing_spec. initialize CorrJob again"
        # Now we want to calculate delays for each pointing & each station & each gate.

        delays_per_station_per_pointing = np.zeros(n_freq * n_time)
        for jjpointing in range(len(self.pointing_spec)):
            delays_all_stations = self.pycalc_results.interpolate_delays(
                Time(tij_unix[:, jjpointing, :].flatten(), format="unix")
            )[:, 0, :, jjpointing]
            # delays_all_stations.shape = (n_freq * n_time,n_station) ; microseconds
            ref_delays_all_stations = (
                delays_all_stations - delays_all_stations[:, iiref, None]
            )
            # subtract delay at the reference station...
            # ...giving us instantaneous baseline delays of shape (n_freq * n_time, n_station) in microseconds
            for iitel, telescope in enumerate(self.telescopes):
                tau_ij = delays_all_stations[:, iitel].reshape((n_freq, n_time))
                tij_unix_sp_coarse[iitel, :, jjpointing, :] = (
                    tij_unix[:, jjpointing, :] + tau_ij * 1e-6
                )  # convert microseconds

        tij_frame_sp = np.zeros(tij_unix_sp_coarse.shape, dtype=int)
        # For each station, start with the coarse float64...
        # ..then use each BBData's precise timing to get the start time aligned to each station's start time.
        for iitel, telescope, bbdata in zip(
            np.arange(n_tel), self.telescopes, self.bbdatas
        ):
            # start with a coarse astropy.Time for this station's time...
            tij_atime_this_station_coarse = Time(
                tij_unix_sp_coarse[iitel],
                val2=np.zeros_like(tij_unix_sp_coarse[iitel]),
                format="unix",
            )
            # ...then translate to frames; use integer frame arithmetic to get the ctime right to nanosecond precision.
            tij_frame_sp[iitel] = self.atime2frame(
                tij_atime_this_station_coarse, bbdata=bbdata, force=True
            )  # yes, lets use a different station; 

            # finally, Astropy arithmetic converts to ctime & offset
            tij_ctime_sp[iitel], tij_ctimeo_sp[iitel] = atime2ctimeo(self.frame2atime(tij_frame_sp[iitel],bbdata=bbdata,force=True))
        return tij_ctime_sp, tij_ctimeo_sp, tij_frame_sp

    def run_correlator_job(
        self,
        event_id,
        gate_spec,
        max_lag=100,
        out_h5_file=None,
        auto_corr = True,
        cross_corr = True,
        clear_bbdata = True,
    ):
        """Run auto- and cross- correlations.
        All BBData are read in at once and ordered using fill_waterfall.
        Loops over stations, does autos.
        Then loops over stations, does fringestopping.
        Then loops over outrigger stations only, cross-correlates baselines.

        This works well on short baseband dumps, but is quite costly RAM-wise since everything is read in at the beginning.

        Parameters
        ----------
        event_id : int
            For writing VLBIVis metadata.
        gate_spec : np.ndarray
            Of start times as a function of (n_freq, n_pointing, n_time)
        max_lag : int
            Maximum lag.
        out_h5_file : string
            Absolute path to .h5 including the extension
        auto_corr: bool
            If True, calculate all autocorrelations. If the empty list is passed, will skip.
        cross_corr : bool
            If True, calculate all cross-correlations. If the empty list is passed, will skip.
        """


        tij_ctime, tij_ctime_offset, tij_frame = self.tij_other_stations(
            gate_spec=gate_spec)

        ref_index = self.ref_index
        bbdata_ref = self.bbdatas[self.ref_index]
        tij_frame_top = tij_frame[self.ref_index]
        n_pointings = bbdata_ref["tiedbeam_baseband"].shape[1] // 2
        n_scan = np.size(tij_frame_top, axis=-1)
        n_pol = 2
        w_ij = gate_spec["duration_frames"][0]  # (npointing, nscan)
        r_ij = gate_spec["dur_ratio"]  # (nfreq, npointing, nscan)
        dm = self.pointing_spec["dm_correlator"]
        output = VLBIVis()
        if auto_corr == True: # do all stations
            auto_corr = np.arange(len(self.bbdatas))
        assert max(auto_corr) < len(self.bbdatas)

        if cross_corr == True: # do all baselines with CHIME x outrigger station, enumerated by their pycalc index
            cross_corr = np.arange(1,len(self.bbdatas))
        assert max(cross_corr) < len(self.bbdatas)

        stations_to_fringestop = set(cross_corr) # from cross_corr, figure out which stations need to be fringestopped
        stations_to_fringestop.add(ref_index) # also need to fringestop ref station
        stations_to_fringestop = list(stations_to_fringestop)
        stations_to_fringestop.sort()

        for iistation in auto_corr:
            this_station = self.telescopes[iistation]
            logging.info(f'Autos for {this_station}')
            bbdata_a = self.bbdatas[iistation]
            gate_this_station = np.empty(gate_spec.shape,dtype = gate_spec.dtype)
            gate_this_station['gate_start_frame'] = tij_frame[iistation]
            gate_this_station['gate_start_unix'] = tij_ctime[iistation]
            gate_this_station['gate_start_unix_offset'] = tij_ctime_offset[iistation]
            gate_this_station['duration_frames'] = gate_spec['duration_frames'] 
            gate_this_station['dur_ratio'] = gate_spec['dur_ratio']
            tij_frame_this_station = tij_frame[iistation]
            # there are scans with missing data: check the start and end index
            # ...but we just let the correlator correlate
            auto_mask = (tij_frame_this_station < 0) + (
                tij_frame_this_station + w_ij > bbdata_a.ntime
            )
            np.clip(
                tij_frame_this_station,
                0,
                bbdata_a.ntime - w_ij,
                out=tij_frame_this_station,
            )
            logging.info(
                f"Calculating autos for station {iistation}; {np.sum(auto_mask)}/{auto_mask.size} scans out of bounds"
            )
            auto_vis = autocorr_core(
                DM=dm,
                bbdata_a=bbdata_a,
                t_a=tij_frame_this_station,
                window=w_ij,
                R=r_ij,
                max_lag=self.max_lag,
                n_pol=2,
            )

            # ...and replace with nans afterward.
            #auto_vis = auto_vis + (auto_mask[:, :, None, None, None, :] * np.nan)
            output._from_ndarray_station(
                event_id,
                telescope=this_station,
                pointing_spec=self.pointing_spec,
                bbdata=bbdata_a,
                auto=auto_vis,
                gate_spec = gate_this_station,
            )
            logging.info(f"Wrote autos for station {iistation}")
            del auto_vis # save memory
        
        # fringestop all relevant stations
        fringestopped_stations = np.zeros((len(stations_to_fringestop), 1024, n_pointings * 2, n_scan, np.max(w_ij.flatten())),dtype = self.bbdatas[0]['tiedbeam_baseband'][:].dtype)
        for iistation in stations_to_fringestop:
            self.bbdatas[iistation]["tiedbeam_baseband"][:] = np.nan_to_num(
                self.bbdatas[iistation]["tiedbeam_baseband"][:], nan=0, posinf=0, neginf=0
            )
            fringestopped_stations[iistation] = fringestop_station(
                bbdata=self.bbdatas[iistation],
                bbdata_top=bbdata_ref,
                pointing_spec = self.pointing_spec,
                t_a=tij_frame_top,
                window=w_ij,
                R=r_ij,
                pycalc_results=self.pycalc_results,
                station_index=iistation,
                ref_frame=ref_index,
                assign_pointing='1to1' # use 1to1 for DM refinement trials or calibrator survey; use 'nearest' for correlator-repointing run.
            ) # bbdata_fs.shape = (nfreq, npointing, nscan, scan_width)
        
        # correlate all N^2 baselines
        out_cross = [] 
        for iioutrigger in cross_corr: # 
            f0 = bbdata_ref.index_map["freq"]["centre"]  # shape is (nfreq)
            assert (f0 == self.bbdatas[iioutrigger].index_map['freq']['centre']).all(), f"Mismatched frequencies between station 0 & station {iioutrigger}! Run fill_waterfall() on both to fix"
            vis_shape = (bbdata_ref.nfreq, n_pointings, n_pol, n_pol, 2 * max_lag + 1, n_scan)
            # loops over scans are within crosscorr_core
            cross = crosscorr_core(
                bbdata_a_fs=fringestopped_stations[ref_index],
                bbdata_b_fs=fringestopped_stations[iioutrigger],
                window=w_ij,
                R=r_ij,
                f0=f0,
                DM=dm,
                index_A=ref_index,
                index_B=iioutrigger,
                max_lag=self.max_lag,
                ref_frame=ref_index,
            )
            tij_ctime_a = tij_ctime[ref_index]  # extract start frame for station
            tij_ctime_b = tij_ctime[iioutrigger]  # extract start frame for station
            avg_ctime = (tij_ctime[ref_index] + tij_ctime[iioutrigger]) * 0.5
            avg_ctimeo = (tij_ctime_offset[ref_index] + tij_ctime_offset[iioutrigger]) * 0.5
            gate_this_baseline = np.empty(gate_spec.shape,dtype = gate_spec.dtype)
            gate_this_baseline['gate_start_unix'] = avg_ctime
            gate_this_baseline['gate_start_unix_offset'] = avg_ctimeo
            gate_this_baseline['duration_frames'] = gate_spec['duration_frames']
            gate_this_baseline['dur_ratio'] = gate_spec['dur_ratio']
            gate_this_baseline['duration_frames'] = -1000000000000 # no meaningful frame count for baseline group! sentinel value should preclude use in most scenarios
            output._from_ndarray_baseline(
                event_id=event_id,
                pointing_spec=self.pointing_spec,
                telescope_a=self.telescopes[ref_index],
                telescope_b=self.telescopes[iioutrigger],
                cross=cross,
                gate_spec = gate_this_station,
            )

            logging.info(f"Wrote visibilities for baseline {self.tel_names[ref_index]}-{self.tel_names[iioutrigger]}")
            del cross
        if clear_bbdata:
            del self.bbdatas # free up space in memory
            del fringestopped_stations

        if type(out_h5_file) is str:
            output.save(out_h5_file)
            logging.info(f"Wrote visibilities to disk: ls -l {out_h5_file}")
        return output
