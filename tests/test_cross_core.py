#import pytest
import numpy as np
import os
from pyfx.core_correlation import crosscorr_core, autocorr_core
from pyfx.core_correlation_station import cross_correlate_baselines,autocorr_core
from pyfx.core_vis import extract_subframe_delay, extract_frame_delay
from pyfx import corr_job_station as corr_job_station

from baseband_analysis.core.sampling import fill_waterfall
from baseband_analysis.core.dedispersion import coherent_dedisp
import astropy.coordinates as ac
from baseband_analysis.core.bbdata import BBData
import astropy.units as un
from astropy.time import Time
import scipy
from pycalc11 import Calc
import logging


chime = ac.EarthLocation.from_geocentric(
    x = -2059166.313 * un.m,
    y = -3621302.972 * un.m,
    z =  4814304.113 * un.m)
chime.info.name = 'chime'

kko = ac.EarthLocation.from_geocentric(
    x = (-2111738.254-10.283) * un.m,
    y = (-3581458.222+4.515) * un.m, 
    z = (4821611.987-3.078) * un.m) 
telescopes = [chime,kko]
kko.info.name = 'kko'

class VeryBasicBBData:
    def __init__(self,
        freq_ids,
        data,
        #ctime,
        #ctime_offset
        ):
        self.data = data
        #self.ctime = ctime
        #self.ctime_offset = ctime_offset
        self.im_freq = np.empty(1024,dtype = [('centre', '<f8'), ('id', '<u4')])
        self.im_freq['centre'] = np.linspace(800,400,num = 1024, endpoint = False)
        self.im_freq['id'] = np.arange(1024)
        self.index_map = {"freq" : self.im_freq[freq_ids]}
        self.nfreq = len(freq_ids)
    def __getitem__(self, key):
        if key == "tiedbeam_baseband":
            return self.data
        if key == "time0": 
            return {
                "ctime": self.ctime, 
                "ctime_offset": self.ctime_offset
            }
        
def dont_test_autocorr_sim(): # deprecated test, renamed from test_autocorr_sim
    """Tests whether output of autocorr makes sense given "simulated" input data.
    Autocorrs should the same in both pols if the input is the same.
    Autocorrs should be the same with zp = False and zp = True (an optimization done around June 2023)
    """
    ntime=100
    nfreq = 9
    freq_ids_present =np.arange(nfreq)
    npol=2
    nscan=1
    npointing=1
    t_a=np.random.randint(0,10,size=(nfreq,npointing,nscan))*0+10
    data=(np.random.uniform(-5e6,5e6,(nfreq,npol*npointing,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,npol*npointing,ntime)))*0
    amplitude_00=8+0j
    amplitude_11=50+0j
    for iifreq in range(nfreq):
        for scan in range(nscan):
            data[iifreq,scan*2,t_a[iifreq,scan]+1]=2+2j #peakval 8
            data[iifreq,scan*2+1,t_a[iifreq,scan]+1]=5+5j #peakval 50
    bbdata_a=VeryBasicBBData(freq_ids=freq_ids_present,data=data)
    R=np.ones((nfreq,npointing,nscan))
    max_lag=10
    window=np.ones((npointing, nscan))*40
    vectorized_autocorr=autocorr_core(DM = 1.2,bbdata_a=bbdata_a, t_a=t_a,window=window,R=R,max_lag=max_lag,zp=False)
    assert np.isclose(vectorized_autocorr[...,:,0,0,0,0],amplitude_00).all()
    assert np.isclose(vectorized_autocorr[...,:,1,1,0,0],amplitude_11).all()
    
    vectorized_autocorr_zp=autocorr_core(DM = 1.2,bbdata_a=bbdata_a, t_a=t_a,window=window,R=R,max_lag=max_lag,zp=True)
    assert np.isclose(vectorized_autocorr,vectorized_autocorr_zp).all(),"autocorr zp does not work"

def test_continuum_calibrator():
    """Tests whether cross correlation of a continuum source yields expected results based on real data.
    Run this on CANFAR in a container containing pycalc, pyfx, and baseband-analysis.
    """
    from outriggers_vlbi_pipeline.vlbi_pipeline_config import gbo
    telescopes = [chime,kko,gbo]
    telescope_names=['chime','kko']
    chime_file='/arc/projects/chime_frb/pyfx_test_files/J0117+8928_chime.h5' 
    kko_file='/arc/projects/chime_frb/pyfx_test_files/J0117+8928_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    
    ra_telA = np.array(chime_bbdata["tiedbeam_locations"]["ra"][0])
    dec_telA = np.array(chime_bbdata["tiedbeam_locations"]["dec"][0])
    ra_telB = np.array(out_bbdata["tiedbeam_locations"]["ra"][0])
    dec_telB = np.array(out_bbdata["tiedbeam_locations"]["dec"][0])
    assert np.isclose(
        ra_telA, out_bbdata["tiedbeam_locations"]["ra"][0], rtol=5 / 60
    ), "tiedbeam pointings are not the same"  # should be consistent to 5 arcmin
    assert np.isclose(
        dec_telA, out_bbdata["tiedbeam_locations"]["dec"][0], rtol=5 / 60
    ), "tiedbeam pointings are not the same"  # should be consistent to 5 arcmin

    ra = np.atleast_1d(ra_telA)
    dec = np.atleast_1d(dec_telA)

    time0 = Time(
        chime_bbdata["time0"]["ctime"],
        val2=chime_bbdata["time0"]["ctime_offset"],
        format="unix",
        precision=9,
    )
    sweep_duration_frames = (
        chime_bbdata["time0"]["fpga_count"][-1] - chime_bbdata["time0"]["fpga_count"][0]
    )
    duration_sec = int(
        2.56e-6 * (chime_bbdata.ntime + sweep_duration_frames) + 50
    )
    nscan=1
    npointing=1
    max_lag=100
    t_a=np.zeros((1024,npointing,nscan),int)#+10000
    R=np.ones((1024,npointing,nscan),int)
    window=np.ones((npointing,nscan),int)

    ntime=int(len(chime_bbdata["tiedbeam_baseband"][nscan][0]))
    window *= ntime#-10000  # set to 1000 for smaller test, max 43670
    
    weight=None

    buffer_seconds=1
    good_indices=[np.where(np.max(chime_bbdata['tiedbeam_baseband'][:,0,:],axis=-1)!=0.0)[0]] #these have not been waterfall filled
    real_times = Time(
        chime_bbdata["time0"]["ctime"][good_indices]-buffer_seconds,
        val2=chime_bbdata["time0"]["ctime_offset"][good_indices],
        format="unix",
        precision=9,
    )
    start_time=np.min(real_times)

    srcs = ac.SkyCoord(
        ra=np.array([ra]),
        dec=np.array([dec]),
        unit='deg',
        frame='icrs',
    )
    ci = Calc(
        station_names=telescope_names,
        station_coords=telescopes,
        source_coords=srcs,
        start_time=start_time,
        duration_min=1,
        base_mode='geocenter', 
        dry_atm=True, 
        wet_atm=True,
        d_interval=1,
    )
    ci.run_driver()
    cross=cross_correlate_baselines(bbdata_top=chime_bbdata, bbdatas=[chime_bbdata,out_bbdata], t_a=t_a, window=window, R=R, pycalc_results=ci,DM=0,
                       station_indices=[0,1],sample_rate=2.56,max_lag=max_lag,n_pol=2,ref_frame=0,
                        weight=weight,fast=True)[0]

    ### rfi flagging
    cutoff_00=np.median(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))
    cutoff_11=np.median(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))
    for iifreq in range(len(cross)):
        val_00=np.median(np.abs(cross[iifreq,0,0,0,:,0])**2,axis=-1)
        val_11=np.median(np.abs(cross[iifreq,0,1,1,:,0])**2,axis=-1)
        if val_00 > cutoff_00:
            cross[iifreq,0,0,0,:,0] *=0
        if val_11 > cutoff_11:
            cross[iifreq,0,1,1,:,0] *=0        

    peaklags= extract_frame_delay(
            cross[:,0,:,:,:,0])  
    peaklag_00=peaklags[0]
    peaklag_11=peaklags[1]

    assert peaklag_00 == 0, "frame lag nonzero!"
    assert peaklag_11 == 0, "frame lag nonzero!"

    delays, snrs = extract_subframe_delay(cross[:,0,:,:,:,0])
    print('test_continuum_calibrator() snr:',snrs)
    print('test_continuum_calibrator() delays:',delays)

    assert np.isclose(delays[0,0],-0.2525,rtol=1e-05), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.25109375,rtol=1e-05), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=70, f"fringe signal to noise is below expected value in 0,0 pol, expected (70,54), got ({snrs[0,0]},{snrs[1,1]})"
    assert snrs[1,1]>=54, f"fringe signal to noise is below expected value in 1,1 pol,expected (70,54), got ({snrs[0,0]},{snrs[1,1]})"



def test_pulsar_core():
    """Tests whether cross correlation of a pulsar yields expected results based on real data using pycalc in crosscorr_core. 
    Run this on CANFAR in a container containing pycalc, pyfx, and baseband-analysis.
    """
    telescopes = [chime,kko]
    telescope_names=['chime','kko']
    chime_file='/arc/projects/chime_frb/pyfx_test_files/304050301_target_B0355+54_chime.h5'
    kko_file='/arc/projects/chime_frb/pyfx_test_files/304050301_target_B0355+54_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra=np.atleast_1d(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.atleast_1d(chime_bbdata['tiedbeam_locations']['dec'][0])

    DM=57.1

    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]
    print('sources:',sources)
    time0 = Time(
        chime_bbdata["time0"]["ctime"],
        val2=chime_bbdata["time0"]["ctime_offset"],
        format="unix",
        precision=9,
    )
    nscan=1
    npointing=1
    max_lag=100

    t_a=np.ones((1024,nscan,npointing),int)*25310
    R=np.ones((1024,nscan,npointing),float)
    window=np.ones((nscan,npointing),int)
    window*=761
    weight=None
    buffer_seconds=1
    good_indices=[np.where(np.max(chime_bbdata['tiedbeam_baseband'][:,0,:],axis=-1)!=0.0)[0]] #these have not been waterfall filled
    real_times = Time(
        chime_bbdata["time0"]["ctime"][good_indices]-buffer_seconds,
        val2=chime_bbdata["time0"]["ctime_offset"][good_indices],
        format="unix",
        precision=9,
    )
    start_time=np.min(real_times)

    srcs = ac.SkyCoord(
        ra=np.array([ra]),
        dec=np.array([dec]),
        unit='deg',
        frame='icrs',
    )
    ci = Calc(
        station_names=telescope_names,
        station_coords=telescopes,
        source_coords=srcs,
        start_time=start_time,
        duration_min=1,
        base_mode='geocenter', 
        dry_atm=False, 
        wet_atm=False,
        d_interval=1,
    )
    ci.run_driver()
    cross=crosscorr_core(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, pycalc_results=ci,DM=57.1,
                        index_A=0, index_B=1,sample_rate=2.56,max_lag=max_lag,n_pol=2,#ref_frame=0,#ref_frame=0,
                        weight=weight)
    cross_copy = cross.copy()
    ### rfi flagging
    cutoff_00=np.median(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))
    cutoff_11=np.median(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))
    for ifreq in range(len(cross)):
        val_00=np.median(np.abs(cross[ifreq,0,0,0,:,0])**2,axis=-1)
        val_11=np.median(np.abs(cross[ifreq,0,1,1,:,0])**2,axis=-1)
        if val_00 > cutoff_00:
            cross[ifreq,0,0,0,:,0] *=0
        if val_11 > cutoff_11:
            cross[ifreq,0,1,1,:,0] *=0        

    peaklags= extract_frame_delay(
            cross[:,0,:,:,:,0])  
    peaklag_00=peaklags[0]
    peaklag_11=peaklags[1]
    assert peaklag_00 == 0, "frame lag nonzero!"
    assert peaklag_11 == 0, "frame lag nonzero!"

    delays, snrs = extract_subframe_delay(cross[:,0,:,:,:,0])
    print('test_pulsar_pycalc() snr:',snrs)
    print('test_pulsar_pycalc() delays:',delays)
    assert snrs[0,0]>=39, f"fringe signal to noise is below expected value in 0,0 pol, expected (39,30), got {snrs[0,0]}"
    assert snrs[1,1]>=30, f"fringe signal to noise is below expected value in 1,1 pol,expected (39,30), got {snrs[1,1]}"
    assert np.isclose(delays[0,0],-0.21765625,rtol=1e-04), f"delays[0,0] wrong! Delays evaluated to be {delays}" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.21578125,rtol=1e-04), f"delays[1,1] wrong! Delays evaluated to be {delays}" #should be good to sub nanosecond

def test_pulsar_pycalc_corrjob():
    """Tests whether cross correlation of a pulsar yields expected results based on real data using pycalc in crosscorr_core. 
    Same as test_pulsar_pycalc() but using pycalc instead of difxcalc-wrapper in CorrJob
    Run this on CANFAR in a container containing pycalc, pyfx, and baseband-analysis.
    """
    from coda.core import VLBIVis
    telescopes = [chime,kko]
    chime_file='/arc/projects/chime_frb/pyfx_test_files/304050301_target_B0355+54_chime.h5'
    kko_file='/arc/projects/chime_frb/pyfx_test_files/304050301_target_B0355+54_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra=np.atleast_1d(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.atleast_1d(chime_bbdata['tiedbeam_locations']['dec'][0])
    print('ra,dec:',ra,dec)
    pointing_spec = np.empty((1,),dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = ra
    pointing_spec['corr_dec'][:] = dec
    pointing_spec['source_name'][:] = 'B0355+54_pytest'
    pointing_spec['dm_correlator'][:] = 57.1
    pulsar_job = corr_job_station.CorrJob(
        bbdatas = [chime_bbdata,out_bbdata],
        telescopes = telescopes,
        pointing_spec = pointing_spec,
	   )
    toa = Time(
        out_bbdata["time0"]["ctime"][0],
        val2=out_bbdata["time0"]["ctime_offset"][0],
        format="unix",
        precision=9,
    ) +25310*2.56e-6*un.s

    gate_spec = pulsar_job.define_scan_params_transient(
			      start_or_toa = 'start',
			      t0f0 = (toa, 800.0),#(1689783027.6518016, 800.0),
			      freq_offset_mode = 'dm',
			      window = [761],
                  period_frames = np.array([1000],dtype=int),
			      r_ij = np.ones(1024) * 1,
                  num_scans_before = 0,
                  num_scans_after = 0
			      )
    vlbivis = pulsar_job.run_correlator_job(
        event_id = 304050301,
        gate_spec = gate_spec, 
        pointing_spec = pointing_spec, 
        max_lag=100,

		out_h5_file = None)
    cross = vlbivis['chime-kko']['vis'][:]
    assert cross.shape == (1024,1,2,2,201,1)
    cutoff_00=np.median(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))
    cutoff_11=np.median(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))
    for ifreq in range(len(cross)):
        val_00=np.median(np.abs(cross[ifreq,0,0,0,:,0])**2,axis=-1)
        val_11=np.median(np.abs(cross[ifreq,0,1,1,:,0])**2,axis=-1)
        if val_00 > cutoff_00:
            cross[ifreq,0,0,0,:,0] *=0
        if val_11 > cutoff_11:
            cross[ifreq,0,1,1,:,0] *=0        
    cross_pycalc=cross
    peaklags= extract_frame_delay(
            cross_pycalc[:,0,:,:,:,0])
    peaklag_00=peaklags[0]
    peaklag_11=peaklags[1]

    assert peaklag_00 == 0, "frame lag nonzero!"
    assert peaklag_11 == 0, "frame lag nonzero!"

    delays_pycalc, snrs_pycalc = extract_subframe_delay(cross_pycalc[:,0,:,:,:,0])
    print('test_pulsar_pycalc_corrjob() snr: ',snrs_pycalc)
    
    # note that the S/N are lower and delays are different from test_pulsar_pycalc
    # this is because the RFI mask zaps a slightly different set of channels
    # delay is _still_ good to sub nanosecond
    logging.info(snrs_pycalc)
    logging.info(delays_pycalc)
    assert np.isclose(delays_pycalc[0,0],-0.21765625,atol=1e-03), f"delays[0,0] wrong! Got {delays_pycalc[0,0]}!"
    assert np.isclose(delays_pycalc[1,1],-0.21578125,atol=1e-03), f"delays[1,1] wrong! Got {delays_pycalc[1,1]}!"
    
    assert snrs_pycalc[0,0]>=37, f"fringe signal to noise is below expected value in 0,0 pol, got {snrs_pycalc[0,0]}"
    assert snrs_pycalc[1,1]>=27, f"fringe signal to noise is below expected value in 1,1 pol, got {snrs_pycalc[1,1]}"
