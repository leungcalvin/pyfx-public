# PyFX
A Python and HDF5-based VLBI correlator for CHIME Outriggers. It is based on the BBData data format used by CHIME, and uses the delay model difxcalc11 developed for the VLBA.

Natively supports frequency-dependent pulsar gates, and coherent dedispersion.


## Installation
```bash
    git clone https://github.com/leungcalvin/pyfx.git
    cd pyfx
    python setup.py develop # one way, or...
    pip install -e . # ...another way, but editable
```

You will also need `pycalc` and you probably will need `coda`. Use `main` branch for both.

## Usage
You will want to set up a `CorrJob` to handle the correlation over all possible baselines. The `CorrJob` will specify the correlation start and stop times as a function of frequency, time, and pointing at a given reference station in three arrays of shape $(nfreq (fixed to 1024), npointing, ntime)$. In most cases you will have only one pointing, but for widefield VLBI you might want multiple pointings.
1) $t$, a `float64`, which specifies topocentric Unix time 
2) $w$, an `int`, which specifies the total width of the integration (<0.5 seconds)
3) $r$, a `float` between 0-1 which specifies the fraction over the integration to correlate (used in pulsar gating mode), centered on `t + 2.56e-6 * w//2`.

```python
    import numpy as np
    from pyfx import corr_job
    
    chime_file='/home/calvin/public/astro_255011898_multibeam_B0136+57_chime.h5' #a singlebeam file, ignore the name
    kko_file='/home/calvin/public/astro_255011898_multibeam_B0136+57_kko.h5' #a singlebeam file from another station, ignore the name
    pulsar_job = corr_job.CorrJob([chime_file,kko_file], #pass the names into CorrJob
       ras = np.array([24.83225]), # where do we point (J2000)?
       decs = np.array([58.24217194]) # where do we point (J2000)?
	   )
    # 1) define correlation job
    t,w,r = pulsar_job.define_scan_params(ref_station = 'chime',
			      t0f0 = (1670732187.8666291, 800.0), # reference time & frequency in MHz
                  start_or_toa = 'start', # is the reference time a "start time" (left edge) or is it a "toa" (center of the integration)
			      time_spacing = 'even', # evenly spaced integrations as a function of time. Later on, need to implement hardcore polyco gating.
			      period_frames = 2000, # spacing between integrations as a function of time
                  freq_offset_mode = 'bbdata', # how does the integration time change vs frequency? Can either be "bbdata" or "dm"
			      Window = np.ones(1024) * 2000, # duration of each integration in frames
			      r_ij = np.ones(1024) * 1, # duration of the gate applied to each integration
			      dm = 73.81141, # DM used for de-smearing, needs to be good to ~1% unless you're using freq_offset_mode = 'dm'
			      num_scans_before = 0, # how many integrations before the reference time, either an int or 'max'
                  num_scans_after = 2, # how many integrations after the reference time, either an int or 'max'
			      ) # these parameters completely define t,w,r.
    # 2) optional: check that the job has reasonable parameters
    pulsar_job.visualize_twr(chime_file,t[...,0:5],w[...,0:5],r[...,0:5],dm = 73.81141) # does a waterfall plot to double check that you're integrating over the pulse.
    
    # 3) press go
    vlbivis = pulsar_job.run_correlator_job(t[...,0:3],w[0,:,0:3].astype(int),r[...,0:3],dm = 73.81141,
			      out_h5_file = False) # performs the correlations by 1) reading in all frequencies for all stations 2) correlating, and 3) writing out a VLBIVis.
    
    # analysis and calibration of visibilities follows hereafter -- see `coda` repo
```

If you want to be janky you can directly run `crosscorr_core()` or `autocorr_core()` and accomplish similar things without the `CorrJob` layer

# HDF5 Baseband Data Format Specification
An ideal data format to hold baseband data used in `pyfx` would be 1) easily interpretable by end users and manipulated with custom Python 3 analysis tools, and 2) easily used in established VLBI correlators like `DiFX` and `SFXC`. Unfortunately such a data format does not exist. Baseband data produced by the full-array baseband systems on ICE-based telescopes are saved to `.h5` files, which are then processed by offline (and later, real-time) beamformers using CHIME/FRB's `singlebeam`  or `multibeam` formats, whose data ordering reflects CHIME's FX correlator architecture. The format specification for `singlebeam` data as used by `pyfx` is summarized here. 

To open `singlebeam` files one can either use `h5py` directly. We discourage this; instead use something like:

```python
from baseband\_analysis.core import BBData
data_all_freqs = BBData.from_file('/path/to/baseband_EVENTID_*.h5') # to load all frequencies
data_first_beam = BBData.from_file('/path/to/baseband_EVENTID_*.h5',beam_sel = [0,1]) # to load all frequencies, just one dual-pol beam
data_first_three_freqs_explicit = BBData.from_file(['/path/to/baseband_EVENTID_0.h5','/path/to/baseband_EVENTID_1.h5','/path/to/baseband_EVENTID_3.h5']) # to load data from FPGA freq_ids = 0,1,3 explicitly
data_first_three_freqs_implicit = BBData.from_file('/path/to/baseband_EVENTID_*.h5',freq_sel = [0,1,2]) # to load data implicitly from the first three files available
```

As one can see, `caput.memh5` does the I/O management under the hood for us, allowing downselection along arbitrary axes. `BBData.from_file` also:
* Handles the offset encoding of raw baseband data (4 real + 4 imaginary), 
* Metadata which keep track of sign flips in the complex conjugate convention taken by the beamformer upstream, changing the sign convention when the data are loaded into memory.
A complete `singlebeam` file should have data and metadata attributes as described below, and `multibeam` files are quite closely related. \textbf{Bolded} refers to features that do not exist or are irrelevant for \texttt{singlebeam} files, but which would be a natural way to extend the data format for the pulsar beam data.

1) `data.index\_map` : a dictionary-like data structure for users to interpret the axes which exist in the `BBData` dataset. The `BBData` dataset holds `np.ndarrays` of data. Here is a list of axes, and metadata describing them:
	1) Observing Frequency: `data.index_map['freq']`  ($N_{\nu} \leq 1024$): `data.index\_map['freq']['centre']` holds the center frequency of each PFB channel, in MHz. Similarly, `data.index\_map['freq']['id']` Holds the frequency ID of each frequency channel as an integer $k$. The mapping from frequency IDs to frequencies (in MHz) is $\nu_k = 800 - 0.390625k$, for $k = 0\ldots 1023$. Because every channel center and frequency ID is specified, the frequency axis is not assumed to be continuous. Note that we do not have the channel centered at 400.0 MHz.
	2) Telescope array element : `data.index\_map['input']['id']` ($N_e \leq 2048$) holds the serial numbers of each antenna used to form the synthesized beam. This axis is no longer present in beamformed baseband data datasets, but the metadata still exist to inform the end user which antennas were combined into a tied-array beam at each station.  
	3) Polarization/Pointing : ($N_p$ assumed to be even): `data.index_map['beam']` is supposed to hold the information about where each station's beams are formed. Currently it just holds integers $0,1,...2n-1$, where $n$ is the number of unique sky locations which are beamformed. The beams and antenna polarization (either 'S' or 'E') are recorded in `data['tiedbeam_locations'][:]`. It is possible to do hundreds of pointings offline in multiple phase center mode in the beamformer, limited only by the size of the file produced per frequency. When $N_p = 2$, we refer to this as a `singlebeam` file whereas when `N_p > 2` we call it a `multibeam` file; the `multibeam` files are typically broken down along the frequency axis to reduce the size of each file.
	4) Time ($N_t \sim 10^4$): `data.index_map['time']['offset_fpga']` holds the index of every FPGA frame after `data['time0']['fpga_count']`.  Only one record of the `fpga_offset` is recorded for all frequency channels, since we do not want to record `data.index_map['time']['fpga_offset']` independently for each channel (which would double our data volume). Therefore, for a particular element of baseband data in array of shape `(nfreq, ntime)`, the Unix time at which the `data['tiedbeam_baseband'][k,:,m]` element was recorded is `data.ctime['time0'][k] + 2.56e-6 * data.index_map['time']['fpga_offset'][m]` 
2) `data['tiedbeam_baseband']` : array of shape ($N_{\nu},N_{p}, N_t$)             Holds the actual baseband data in an array of complex numbers. The baseband data should be flux-calibrated such that the mean of the power obtained by squaring the data is in units of Janskys * $f_{good}^2$ here $f_{good}$ is the fraction of antennas that are not flagged. The baseband data have an ambiguous complex conjugate convention. Data that obeys the same complex conjugate convention as raw PFB output from the F-engine also has the attribute \texttt{data[`tiedbeam\_baseband`].attrs[`conjugate\_beamform`] = 1}, whereas data that has the opposite convention (data processed prior to October 2020) lacks this attribute.

3) data['time0'] : array of shape $(N_{\nu})$ \\
        Holds the absolute start time of each baseband dump as a function of frequency channel as a pair of `float64`s, in `data['time0']['ctime']` and `data['time0']['ctime_offset']` respectively. Times are formatted as a UNIX timestamp in seconds (since midnight on January 1 1970 in UTC time). Since the baseband dumps start at a different time in each frequency channel, \texttt{ctime} is recorded as a function of frequency channel, disciplined via a GPS-disciplined crystal oscillator, to the nearest nanosecond. The precision of `ctime` is $\approx \SI{100}{\ns}$ because it is stored as `float64`. Therefore, for most applications using `ctime` alone is sufficient. However, since a `float64` cannot hold UNIX timestamps to nanosecond precision ($\approx$ 19 decimal digits are needed), a second `float64` holds the last few relevant decimal places of the full UNIX time in seconds. Because of the limitations of a `float64` it is often the case that `ctime_offset` is less than several hundreds of nanoseconds. `data['time0']['ctime']` and
        `data['time0']['ctime_offset']` can be easily converted to `astropy.Time` objects using the `val2` keyword. If you do high precision arithmetic, you will find that `ctime` + `ctime_offset` mod 2.56e-6 is a constant over all frequency channels. In addition, `data['time0']['fpga_count']` can be used to calculate the start time of the dump to within a nanosecond. This calculation can be performed for each frequency channel, and the results should be consistent to $10^{-10}$ seconds.
    4) `data['tiedbeam_locations']['ra','dec', or 'pol']` : array of shape $(N_p)$ where $N_p$ is even holds the sky locations and polarizations used to phase up each station. It will also include `data['tiedbeam_locations']['X_400MHz','Y_400MHz']` which refer to local beam-model coordinates done via the `beam_model` package.
    5) `data['centroid']` Holds the position of the telescope's effective centroid, measured from (0,0,0) in local telescope coordinates, in meters,  measured in either a Easting/Northing coordinate system (TONE) or in a $F_\perp,F_\parallel$ coordinate system (perpendicular or parallel to the focal line) as a function of frequency channel. This is a function of frequency because the telescope's centroid is a sensitivity-weighted average of antenna positions (Post-beamforming). This is not yet used in VLBI, but we have the machinery to perform small baseline corrections using this field if necessary.
    6) data['telescope'].attrs['name'] (Not implemented yet?) Holds the name of the station (`chime', `pathfinder', `tone', `allenby', or `greenbank', or `hatcreek')  (Kenzie please update?)
# HDF5 Visibilities Data Format Specification

CHIME Outriggers will have a small number of stations collecting full-array baseband dumps and forming multiple synthesized beams. Since each baseline must be correlated and calibrated independently, we store each baseline and each station as its own independent HDF5 group within a HDF5 container (again inherited from `caput`) called `VLBIVis`. Each station group contains station-related metadata copied from the `singlebeam` data (via `coda.core.VLBIVis.copy_station_metadata`, which copies all attributes stored in the `BBData` to its corresponding station HDF5 group.
The station groups also hold autocorrelation visibilities up to some maximum lag (20 * 2.56 us by default), while each baseline holds per-baseline (e.g. calibration) metadata and cross-correlation visibilities. For example, processing data from CHIME and KKO would result in two autocorrelation HDF5 groups (`vis['chime']`, `vis['tone']`,), and one cross-correlation HDF5 group `vis['chime-kko']` (baselines are alphabetically organized, which has the advantage that `chime` is always station A.

The cross-correlation visibilities, stored in `vis['chime-tone']['vis']` are packed in `np.ndarray`s of shape $(N_\nu, N_{c}, N_{p}, N_{p},N_{\ell},N_t)$. The axes are as follows:

1) $N_b$ denotes the number of baselines. In CHIME Outriggers, we only consider baselines involving CHIME (no outrigger-outrigger baselines) for now. This simplifies the accounting and computation because one never has to compensate each dataset in $N-1$ different ways. 
2) $N_\nu$ enumerates the number of frequency channels. Because fringe-finding involves taking Fourier transforms over the frequency axis, this is fixed to 1024 for now, and infilled with zeros where frequency channels are corrupted by e.g. RFI.
3) $N_{c} \lesssim10$ enumerates the number of correlation phase centers. Usually one or several ($<10$) phase centers will be used per beam, but `difxcalc` can be compiled to supports many ($\approx 250$ before I ran into weird bugs). Currently, we can assign a single (or multiple) VLBI "pointing" to each tied-array "beam" whose width is $0.25 \times 0.25$ degrees, in anticipation of science cases for assigning multiple VLBI pointings per synthesized beam (which a tracking beam may have the sensitivity to see).
4) $N_p \times N_p$ indicates all possible combinations of antenna polarizations. There are two antenna polarizations for each telescope, and they will be labeled ``south'' and ``east'' to denote ``parallel to the cylinder axis'' and ``perpendicular to the cylinder axis'' directions respectively. Since CHIME/FRB Outriggers have co-aligned, dual-polarization antennas, correlating in a linear basis is straightforward and removes the need for polarization calibration.
5) $N_{\ell} \sim 20$ indicates the number of integer time lags saved (in units of $\SI{2.56}{\us}$). In principle, only a few ($<10$) are needed, but it is not difficult to compute and save roughly 20 integer lags, which also allows for some post-correlation frequency upchannelization if desired (e.g. for high rotation measure or diffractive scintillation analysis)
6) $N_{t} \sim 10^{1-4}$ for FRB baseband data enumerates the number of off-pulses correlated in order to estimate the statistical error on the on-pulse visibilities. However, for a 30-second long tracking beam integration with thousands of short pulse windows centered on individual pulsar pulses, $N_{t}$ can approach $\approx 10^4$ for a long pulsar integration.

In addition to the visibilities we also save the following metadata. At the time of cross-correlation, two `singlebeam` (or `multibeam`) files are processed to produce one visibility dataset. In addition to the metadata in both inputted \texttt{singlebeam} files (as described above) we will save...

1) Software metadata -- `github` commit hash denoting what version of the correlator produced the file.
2) `vis['chime']['time_a']`:  The topocentric start time of each integration at each station to nanosecond precision (see `BBData['time0']`) as a function of frequency and time.
3) `vis['chime-tone']['vis'].attrs['station_a','station_b']`: `Astropy.EarthLocation` objects denoting the geocentric `(X,Y,Z)` positions of the stations fed into `difxcalc11`
4) `vis['chime-kko']['vis'].attrs['calibrated']`: a boolean attribute denoting whether phase + delay calibration has been applied to the visibilities via `coda.calibration.apply_phase_cal`.
5) `vis['chime-tone']['vis'].attrs['clock_jitter_corrected']` and `['clock_drift_corrected']` refer to whether one-second timescale clock jitter (between the GPS and maser) has been calibrated out, and weeks-long timescale clock drift (between masers at two stations) has been calibrated out using the CHIME/FRB `maser` pipeline. Use `coda.clock.apply_clock_jitter` and `coda.clock.apply_clock_drift` to apply/unapply these corrections.

# Maintainers/Developers/Sufferers
Calvin Leung
Shion Andrew
