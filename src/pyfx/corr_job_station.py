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

K_DM = 1 / 2.41e-4  # in s MHz^2 / (pc cm^-3)
FREQ = np.linspace(800,400,num = 1024, endpoint = False)

def same_pointing(bbdata_a,bbdata_b):
    assert np.isclose(bbdata_a['tiedbeam_locations']['ra'][:],bbdata_b['tiedbeam_locations']['ra'][:]).all()
    assert np.isclose(bbdata_a['tiedbeam_locations']['dec'][:],bbdata_b['tiedbeam_locations']['dec'][:]).all()
    return True

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
    #assert ((sub_scan_end[:,:,:,iisort][:,:,:,0:-1] - sub_scan_start[:,:,:,iisort][:,:,:,1:]) < w_ij[:,:,:-1][None] * 2.56e-6 * 0.01).all(), "next scan overlaps more than 1% with previous scan? you probably do not want this" 
    
    # no changing integer lags
    earth_rotation_time = 0.4125  # seconds https://www.wolframalpha.com/input?i=1.28+us+*+c+%2F+%28earth+rotation+speed+*+2%29
    freq = np.linspace(800, 400, num=1024, endpoint=False)  # no nyquist freq
    assert np.max(w_ij * 2.56e-6) < earth_rotation_time, "Use smaller value of w_ij, scans are too long!"

    if dm is not None:  # check that wij exceeds smearing time
        dm_smear_sec = K_DM * dm * 0.390625 / freq**3
        diff = np.min(w_ij * 2.56e-6 - dm_smear_sec[:,None,None], axis=(-2,-1)) # check all frequency channels
        if not (diff > 0).all():
            logging.warning(f"For DM = {dm}, w_ij needs to be increased by {-(np.min(diff)/ 2.56e-6):0.2f} frames to not clip the pulse within a channel")
    return w_ij

class CorrJob:
    
    correlator_gating_dtypes = {
            ("gate_start_unix", "<f8"),
            ("gate_start_unix_offset", "<f8"),
            ("gate_start_frame", "<i4"),
            ("duration_frames", "<i4"),
            ("dur_ratio", "<f8"),
            ("on_window", bool),
    }
    def __init__(
        self, 
        bbdatas: List[BBData],
        telescopes:List[ac.earth.EarthLocation], 
        bbdata_filepaths:Optional[List[str]]=None, 
        ref_station:Optional[str]='chime',
        ras = None, 
        decs = None,
        source_names=None):
        """Set up the correlation job:
        Get stations and order the bbdata_list as expected by difxcalc.
        Run difxcalc with a single pointing center.
        Choose station to use as the reference station, at which t_{ij} is initially inputted.
        For each station, define t_ij, w_ij, and r_ij into arrays of shape (N_baseline, N_freq, N_time) by calling define_scan_params().
        Given a set of BBData objects, define N * (N-1) / 2 baselines.
        Use run_difxcalc and save to self.pycalc_results so we only call difxcalc ONCE in the whole correlator job.
        """
        self.station_names= []
        self.bbdatas = bbdatas
        # get tel names
        for i,this_bbdata in enumerate(bbdatas):
            tel_name=station_from_bbdata(
                    this_bbdata
                    )
            self.station_names.append(
                    tel_name
                )
            if tel_name != telescopes[i].info.name:
                print(f"warning: telescope name {telescopes[i].info.name} from input telescopes does not correspond to telescope name {tel_name} from bbdata. Please check that your input parameters are identically ordered.")
        self.telescopes = telescopes 
        self.ref_station=ref_station
        ref_index=self.station_names.index(ref_station)
        self.ref_index=ref_index
        bbdata_top = self.bbdatas[self.ref_index]

        earliest_start_unix = np.inf
        latest_end_unix = -np.inf

        # loop again to check data format specifications
        for i,this_bbdata in enumerate(bbdatas):
            assert same_pointing(bbdata_top,this_bbdata)
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
        # Get pointing centers from reference station, if needed.
        if ras is None:
            ras = bbdata_top['tiedbeam_locations']['ra'][:]
        if decs is None:
            decs = bbdata_top['tiedbeam_locations']['dec'][:]
        if source_names is None:
            source_names = bbdata_top['tiedbeam_locations']['source_name'][:]
        
        self.ras = np.atleast_1d(ras)
        self.decs = np.atleast_1d(decs)
        self.source_names = np.atleast_1d(source_names)
        assert len(ras)==len(decs), "number of pointings is not consistent between ras and decs!"
        assert len(ras)==len(source_names), "number of pointings is not consistent between ras and source_names!"
        self.pointings = ac.SkyCoord(ra=self.ras.flatten() * un.deg, dec=self.decs.flatten() * un.deg)

        ci = Calc(
                station_names=self.station_names,
                station_coords=self.telescopes,
                source_coords=self.pointings,
                start_time=Time(np.floor(earliest_start_unix), format = 'unix', precision = 9),
                duration_min=duration_min,
                base_mode='geocenter', 
                dry_atm=True, 
                wet_atm=True,
                d_interval=1,
            )
        ci.run_driver()
        self.pycalc_results=ci
        return 


    def t0_f0_from_bbdata_filename(self,
        t0f0:Tuple[str],
        bbdata_ref_filename:str
        ):
        """ Returns the actual t00 and f0 from bbdata_ref.

        This allows you to specify, e.g. the "start" of the dump at the "top" of the band by passing in ("start","top").
        t0f0: tuple specifying start/middle of the dump in time and top/bottom of the frequency band
        bbdata_ref_filename: name of filename
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
            logging.info('INFO: Offset between requested frequency and closest frequency: {offset_mhz} MHz')
        f0 = im_freq['centre'][iifreq] # the actual reference frequency.

        if _t0 == 'start':
            t0 = time0['ctime'][iifreq]  # the actual reference start time
        if _t0 == 'middle':
            ntime = get_ntime(bbdata_ref_filename)
            t0= time0['ctime'][iifreq] + ntime * 2.56e-6 // 2
        return t0,f0
        
    def define_scan_params(
        self,
        dm:float,
        t_ij:Optional[np.ndarray]=None,
        t0f0:Optional[Tuple[float,float]]=None,
        start_or_toa = 'start',
        freq_offset_mode = 'bbdata',
        Window = np.ones(1024,dtype = int) * 1000,
        r_ij = 1.0,
        max_lag = 100,
    ):
        """
        Tells the correlator when to start integrating, how long to start integrating, for each station. Run this after the CorrJob() is instantiated.

        t0f0 : tuple consiting of two floats
            First float: denotes the start time of the integration with a unix time 
            Second float: denotes frequency channel
        start_or_toa : 'start' or 'toa'
            Interpret the given time as a start time or a "center" time.

        freq_offset_mode : 'bbdata', or 'dm'
        
        width : 'fixed', 'from_time'
        
        Window : np.ndarray
            Sets the integration duration in frames as a function of frequency.
        dm : float
            A dispersion measure for gating. Get this right to the 2-3rd decimal place.
        
        kwargs : 'dm' and 'f0', 'pdot','wi'
        """

        bbdata_ref_filename = self.bbdata_filepaths[self.tel_names.index(self.ref_station)]

        
        # First do t_i0 for the reference station...
        if freq_offset_mode == "dm":
            assert t0f0 is not None, "t0f0 (unix toa [s], ref freq[Mhz]) must be passed in if freq_offset_mode is dm"
            (t00, f0) = t0f0 #only support ac time and floats for now
            assert isinstance(t00,astropy.time.core.Time), "t00 must be an astropy time"
            t_ij = self._ti0_from_t00_dm(t00, f0, dm = dm, fi = FREQ)
            t_ij=t_ij.reshape(t_ij.shape[0],1,1) #only supports single pointing single scan in this mode right now
        else:
            assert t_ij is not None, "t_ij (np.ndarray of start indices of shape [nfreq,npointing,nscan]) must be passed in if freq_offset_mode is not dm!"
        

        # If _ti0 is a TOA, need to shift _ti0 back by half a scan length 
        if start_or_toa == 'toa':
            t_ij -= Window/2  # Scan duration given by :Window:, not :period_frames:!
            logging.info('INFO: Using TOA mode: shifting all scans to be centered on t_ij')

        ctime = Time(
            self.ref_ctimes[0],
            val2=self.ref_ctime_offsets[0],
            format="unix",
            precision=9,
        )
        tij_unix=t_ij*2.56e-6*un.s+ctime
        assert Window.shape[0] == 1024, "Need to pass in the length of the integration as a function of frequency channel!"

        logging.info('Success: generated a valid set of integrations! Now call run_correlator_job() or run_correlator_job_multiprocessing()')
        t_ij_station_pointing = self.tij_other_stations(t_ij, tij_unix=tij_unix)
        
        # Check that the time spacing works.
        if Window.ndim == 1: # broadcast to the shape of tij
            Window = Window[:,None,None] + np.zeros_like(t_ij_station_pointing[0])
        if r_ij.ndim == 1: # broadcast to the shape of tij
            r_ij = r_ij[:,None,None] + np.zeros_like(t_ij_station_pointing[0])
        
        validate_wij(Window,t_ij_station_pointing, r_ij, dm = dm)

        self.max_lag = max_lag
        return t_ij_station_pointing,  Window.astype(int), r_ij
    
    def round_to_integer_frame(
        self,
        timestamps: np.ndarray):
        
        ctime = Time(
                    self.ref_ctimes,
                    val2=self.ref_ctime_offsets,
                    format="unix",
                    precision=9,
                )
        closest_frame=np.round(((timestamps-ctime).sec/(2.56e-6*un.s)).value).astype(int)
        #timestamps_rounded=ctime+closest_frame*2.56e-6*un.s
        return closest_frame #timestamps_rounded


    def _ti0_from_t00_dm(self, t00, f0, dm, fi):
        ti0 = t00 + K_DM * dm * (fi**-2 - f0**-2)*un.s  # for fi = infinity, ti0 < t00.
        return self.round_to_integer_frame(ti0)

    def tij_other_stations(
        self, 
        tij, 
        tij_unix:np.ndarray):
        """Do this on a per-pointing and per-station basis."""
        iiref = self.tel_names.index(self.ref_station)

        import astropy
        tij_sp = np.zeros(
            (len(self.telescopes),
             1024,
            len(self.pointings),
            tij.shape[-1]
            ),
            dtype = int) 

        delays= self.pycalc_results.interpolate_delays(Time(tij_unix.flatten(),format = 'unix'))[:,0,:,:] #delays.shape = (n_freq * n_time,n_station, n_pointing) 
        delays -= delays[:, iiref, None,:] # subtract delay at the reference station -- now we have instantaneous baseline delays of shape (n_freq * n_time, n_station, n_pointing)
        for iitel, telescope in enumerate(self.telescopes):
            for jjpointing, pointing in enumerate(self.pointings):
                tau_ij = delays[:, iitel, jjpointing].reshape(tij[:,jjpointing,:].shape)/2.56 #frames
                tij_sp[iitel,:,jjpointing,:] = tij[:,jjpointing,:] + tau_ij
        self.tij_sp = tij_sp
        return tij_sp


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



    def run_correlator_job(
        self,
        gate_start_frame, 
        w_ij, 
        r_ij, 
        dm = None, 
        event_id = None, 
        out_h5_file = None):
        """Run auto- and cross- correlations.

        Loops over baselines, then frequencies, which are all read in at once and ordered using fill_waterfall. This works well on short baseband dumps. 
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
        """

        ref_index=self.tel_names.index(self.ref_station)
        output = VLBIVis()
        pointing_centers = np.zeros((len(self.pointings),),dtype = output._dataset_dtypes['pointing'])
        pointing_centers['corr_ra'] = self.ras
        pointing_centers['corr_dec'] = self.decs
        pointing_centers['source_name'] = self.source_names
        bbdata_top = BBData.from_file(self.bbdata_filepaths[ref_index])
        fill_waterfall(bbdata_top, write = True)

        tel_bbdatas=[]

        for iia in range(len(self.tel_names)):
            bbdata_a = BBData.from_file(self.bbdata_filepaths[iia])
            fill_waterfall(bbdata_a, write = True)
            tel_bbdatas.append(bbdata_a)
            logging.info(bbdata_a['tiedbeam_baseband'].shape)

            assert np.issubdtype(gate_start_frame.dtype, np.integer), "gate_start_frame must be an integer start frame"
            gate_start_frame_tel = gate_start_frame[iia] #extract start frame for station
            if iia==ref_index:
                gate_start_frame_top=gate_start_frame_tel
            # there are scans with missing data: check the start and end index
            mask_a = (gate_start_frame_tel < 0) + (gate_start_frame_tel  + w_ij[None,:,:] > bbdata_a.ntime) 

            gate_start_frame_tel[mask_a] = int(bbdata_a.ntime // 2)
            # ...but we just let the correlator correlate
            logging.info(f'Calculating autos for station {iia}')
            auto_vis = autocorr_core(DM=dm, bbdata_a=bbdata_a, 
                                    t_a = gate_start_frame_tel,
                                    window = w_ij,
                                    R = r_ij,
                                    max_lag = self.max_lag, 
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
                bbdata_top=bbdata_top,
                t_a_top=gate_start_frame_top,
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
            for telB in range(telA+1,len(tel_bbdatas)):
                output._from_ndarray_baseline(
                    event_id=event_id,
                    pointing_center=pointing_centers,
                    telescope_a=self.telescopes[telA],
                    telescope_b=self.telescopes[telB],
                    cross=cross[m], 
                    t_a=bbdata_top["time0"]["ctime"][:,np.newaxis,np.newaxis]*np.ones(gate_start_frame_tel.shape),
                    t_a_offset=bbdata_top["time0"]["ctime_offset"][:,np.newaxis,np.newaxis]+gate_start_frame_tel*2.56e-6,
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
