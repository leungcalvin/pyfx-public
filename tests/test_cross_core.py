#import pytest
import numpy as np
import os
from pyfx.core_correlation import autocorr_core,crosscorr_core
from pyfx.core_correlation_pycalc import crosscorr_core as crosscorr_core_pycalc
from pyfx.core_vis import extract_subframe_delay, extract_frame_delay
import pyfx.core_correlation as core_correlation

import difxcalc_wrapper.runner as dcr
from difxcalc_wrapper.io import make_calc
from baseband_analysis.core.sampling import fill_waterfall
from baseband_analysis.core.dedispersion import coherent_dedisp
import astropy.coordinates as ac
from baseband_analysis.core.bbdata import BBData
import astropy.units as un
from astropy.time import Time
import scipy
from pycalc11 import Calc

DIFXCALC_CMD = "/lib/difxcalc11/build/bin/difxcalc"
DIFXCALC_SCRATCH_DIR = "/arc/projects/chime_frb/vlbi/scratch"


chime = ac.EarthLocation.from_geocentric(
    x = -2059166.313 * un.m,
    y = -3621302.972 * un.m,
    z =  4814304.113 * un.m)
chime.info.name = 'chime'

kko = ac.EarthLocation.from_geocentric(
    x = (-2111738.254-10.283) * un.m,
    y = (-3581458.222+4.515) * un.m, 
    z = (4821611.987-3.078) * un.m) 
    
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
        
def test_autocorr_sim():
    #tests whether output of autocorr makes sense given "simulated" input data
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
    """Tests whether cross correlation of a continuum source yields expected results based on real data"""
    telescopes = [chime,kko]
    chime_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/J0117+8928_chime.h5'
    kko_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/J0117+8928_kko.h5'
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

    ra = ra_telA
    dec = dec_telA

    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]

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
    calc_filename = f"{chime_bbdata.attrs['event_id']}_{len(sources)}_sources"
    calcfile = os.path.join(DIFXCALC_SCRATCH_DIR, calc_filename + ".calc")
    calc_params = make_calc(
        telescopes=telescopes,
        sources=sources,
        time=min(time0)
        - 20
        * un.s,
        duration_sec=duration_sec,
        ofile_name=calcfile,
    )

    calcresults = dcr.run_difxcalc(
        calcfile,
        sources=sources,
        difxcalc_cmd=DIFXCALC_CMD,
        remove_calcfile=False,
        force=True,
    )

    nscan=1
    npointing=1
    max_lag=100
    t_a=np.zeros((1024,npointing,nscan),int)
    R=np.ones((1024,npointing,nscan),int)
    window=np.ones((npointing,nscan),int)

    ntime=len(chime_bbdata["tiedbeam_baseband"][nscan][0])
    window *= ntime  # set to 1000 for smaller test, max 43670
    
    weight=None
    cross=crosscorr_core(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, calc_results=calcresults,DM=0,
                        index_A=0, index_B=1,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight)

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
    assert np.isclose(delays[0,0],-0.2521875,rtol=1e-05), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.25078125,rtol=1e-05), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=70, "fringe signal to noise is below expected value in 0,0 pol"
    assert snrs[1,1]>=54, "fringe signal to noise is below expected value in 1,1 pol"

def test_continuum_calibrator_pycalc():
    """Tests whether cross correlation of a continuum source yields expected results based on real data.
    Calls version of the code that uses pycalc11. 
    """
    telescopes = [chime,kko]
    telescope_names=['chime','kko']
    chime_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/J0117+8928_chime.h5' 
    kko_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/J0117+8928_kko.h5'
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

    ra = ra_telA
    dec = dec_telA

    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]

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
    t_a=np.zeros((1024,npointing,nscan),int)
    R=np.ones((1024,npointing,nscan),int)
    window=np.ones((npointing,nscan),int)

    ntime=len(chime_bbdata["tiedbeam_baseband"][nscan][0])
    window *= ntime  # set to 1000 for smaller test, max 43670
    
    weight=None
    srcs = ac.SkyCoord(
        ra=ra,
        dec=dec,
        unit='deg',
        frame='icrs',
    )
    source_coords = [srcs]
    source_names = [f"src{si}" for si in range(len(source_coords))]

    start_time = np.min(time0)
    print(start_time-time0[0])
    duration_min = 1

    ci = Calc(
        station_names=telescope_names,
        station_coords=telescopes,
        source_names=source_names,
        source_coords=source_coords,
        time=start_time,
        duration_min=duration_min,
        base_mode='geocenter', 
        dry_atm=False, 
        wet_atm=False
    )
    ci.run_driver()
    cross=crosscorr_core_pycalc(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, pycalc_results=ci,DM=0,
                        index_A=0, index_B=1,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight,fast=True)

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
    assert np.isclose(delays[0,0],-0.2521875,rtol=1e-05), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.25078125,rtol=1e-05), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=70, "fringe signal to noise is below expected value in 0,0 pol"
    assert snrs[1,1]>=54, "fringe signal to noise is below expected value in 1,1 pol"


def test_pulsar():
    #tests whether cross correlation of a pulsar yields expected results based on real data
    telescopes = [chime,kko]
    chime_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_chime.h5'
    kko_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra=chime_bbdata['tiedbeam_locations']['ra'][0]
    dec=chime_bbdata['tiedbeam_locations']['dec'][0]

    DM=57.1

    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]

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
    calc_filename = f"{chime_bbdata.attrs['event_id']}_{len(sources)}_sources"
    calcfile = os.path.join(DIFXCALC_SCRATCH_DIR, calc_filename + ".calc")
    calc_params = make_calc(
        telescopes=telescopes,
        sources=sources,
        time=min(time0)
        - 20
        * un.s,
        duration_sec=duration_sec,
        ofile_name=calcfile,
    )

    calcresults = dcr.run_difxcalc(
        calcfile,
        sources=sources,
        difxcalc_cmd=DIFXCALC_CMD,
        remove_calcfile=False,
        force=True,
    )

    nscan=1
    npointing=1
    max_lag=100

    t_a=np.ones((1024,nscan,npointing),int)*25310
    R=np.ones((1024,nscan,npointing),int)

    window=np.ones((nscan,npointing),int)
    window*=761
    weight=None
    cross=crosscorr_core(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, calc_results=calcresults,DM=DM,
                        index_A=0, index_B=1,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight,fast=True)

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
    assert np.isclose(delays[0,0],-0.21765625,rtol=1e-04), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.21578125,rtol=1e-04), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=39, "fringe signal to noise is below expected value in 0,0 pol"
    assert snrs[1,1]>=30, "fringe signal to noise is below expected value in 0,0 pol"


def test_pulsar_pycalc():
    #tests whether cross correlation of a pulsar yields expected results based on real data
    telescopes = [chime,kko]
    telescope_names=['chime','kko']
    chime_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_chime.h5'
    kko_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra=chime_bbdata['tiedbeam_locations']['ra'][0]
    dec=chime_bbdata['tiedbeam_locations']['dec'][0]

    DM=57.1

    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]

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
    R=np.ones((1024,nscan,npointing),int)

    window=np.ones((nscan,npointing),int)
    window*=761
    weight=None
    srcs = ac.SkyCoord(
        ra=ra,
        dec=dec,
        unit='deg',
        frame='icrs',
    )
    source_coords = [srcs]
    source_names = [f"src{si}" for si in range(len(source_coords))]

    start_time = np.min(time0)
    duration_min = 1

    ci = Calc(
        station_names=telescope_names,
        station_coords=telescopes,
        source_names=source_names,
        source_coords=source_coords,
        time=start_time,
        duration_min=duration_min,
        base_mode='geocenter', 
        dry_atm=False, 
        wet_atm=False
    )
    ci.run_driver()
    cross=crosscorr_core_pycalc(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, pycalc_results=ci,DM=0,
                        index_A=0, index_B=1,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight,fast=True)

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
    assert np.isclose(delays[0,0],-0.21765625,rtol=1e-04), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.21578125,rtol=1e-04), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=39, "fringe signal to noise is below expected value in 0,0 pol"
    assert snrs[1,1]>=30, "fringe signal to noise is below expected value in 0,0 pol"



def test_pulsar_x():
    from pyfx import corr_job
    chime_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_chime.h5'
    kko_file='/arc/projects/chime_frb/shiona/public/pyfx_test_files/304050301_target_B0355+54_kko.h5'

    pulsar_job = corr_job.CorrJob([chime_file,kko_file],
       ras = np.array([ra]),
       decs = np.array([dec])
	   )

    t,w,r = pulsar_job.define_scan_params(ref_station = 'chime',
			      start_or_toa = 'start',
			      t0f0 = (1689783030.0, 800.0),
			      time_spacing = 'even',
			      freq_offset_mode = 'bbdata',
			      Window = np.ones(1024) * 761,
			      r_ij = np.ones(1024) * 1,
			      period_frames = 761,
			      dm = 57.1,
			      num_scans_before = 0,
                              num_scans_after = 2,
			      )

    vlbivis = pulsar_job.run_correlator_job(t[...,0:3],w[0,:,0:3].astype(int),r[...,0:3],dm = 57.1,
			      out_h5_file = False)
    cross = vlbivis['chime-kko']['vis'][:]
    assert cross.shape == (1024,1,2,2,41,3)
    cutoff_00=np.median(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,0,0,:,0])**2,axis=-1))
    cutoff_11=np.median(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))+1*scipy.stats.median_abs_deviation(np.median(np.abs(cross[:,0,1,1,:,0])**2,axis=-1))
    for iifreq in range(len(cross)):
        val_00=np.median(np.abs(cross[iifreq,0,0,0,:,0])**2,axis=-1)
        val_11=np.median(np.abs(cross[iifreq,0,1,1,:,0])**2,axis=-1)
        if val_00 > cutoff_00:
            cross[iifreq,0,0,0,:,0] *=0
            print(iifreq,'zeroed')
        if val_11 > cutoff_11:
            cross[iifreq,0,1,1,:,0] *=0
    peaklags= extract_frame_delay(
            cross[:,0,:,:,:,0])
    peaklag_00=peaklags[0]
    peaklag_11=peaklags[1]

    assert peaklag_00 == 0, "frame lag nonzero!"
    assert peaklag_11 == 0, "frame lag nonzero!"

    delays, snrs = extract_subframe_delay(cross[:,0,:,:,:,0])
    assert np.isclose(delays[0,0],-0.21765625,rtol=1e-04), "delays[0,0] wrong!" #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.21578125,rtol=1e-04), "delays[1,1] wrong!" #should be good to sub nanosecond
    assert snrs[0,0]>=39, "fringe signal to noise is below expected value in 0,0 pol"
    assert snrs[1,1]>=30, "fringe signal to noise is below expected value in 0,0 pol"

