import pytest
import numpy as np
import copy
import os
from pyfx.core_correlation import autocorr_core,crosscorr_core
from pyfx.core_vis import extract_subframe_delay, extract_frame_delay


import difxcalc_wrapper.runner as dcr
from difxcalc_wrapper.io import make_calc
from baseband_analysis.core.sampling import fill_waterfall
import astropy.coordinates as ac
from baseband_analysis.core.bbdata import BBData
import astropy.units as un
from astropy.time import Time
import scipy

DIFXCALC_CMD='/home/calvin/difxcalc11/build/bin/difxcalc'
DIFXCALC_SCRATCH_DIR='/home/calvin/public/'

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
    t_a=np.random.randint(0,10,size=(nfreq,nscan,npointing))*0+10
    data=(np.random.uniform(-5e6,5e6,(nfreq,npol*npointing,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,npol*npointing,ntime)))*0
    amplitude_00=8+0j
    amplitude_11=50+0j
    for iifreq in range(nfreq):
        for scan in range(nscan):
            data[iifreq,scan*2,t_a[iifreq,scan]+1]=2+2j #peakval 8
            data[iifreq,scan*2+1,t_a[iifreq,scan]+1]=5+5j #peakval 50
    bbdata_a=VeryBasicBBData(freq_ids=freq_ids_present,data=data)
    R=np.ones((nfreq,nscan,npointing))
    max_lag=10
    window=np.ones((nscan,npointing))*40
    vectorized_autocorr=autocorr_core(DM = 1.2,bbdata_a=bbdata_a, t_a=t_a,window=window,R=R,max_lag=max_lag)
    assert np.isclose(vectorized_autocorr[...,:,0,0,0,0],amplitude_00).all()
    assert np.isclose(vectorized_autocorr[...,:,1,1,0,0],amplitude_11).all()


"""
def test_crosscorr_sim():
    #tests whether output of crosscorr makes sense given "simulated" input data
"""
def test_continuum_calibrator():
    #tests whether cross correlation of a continuum source yields expected results based on real data
    telescopes = [chime,kko]
    chime_file='/home/calvin/public/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_chime.h5'
    kko_file='/home/calvin/public/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)

    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra= 24.2440833
    dec= 47.8580556
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
    t_a=np.zeros((1024,nscan,npointing),int)
    R=np.ones((1024,nscan,npointing),int)
    window=np.ones((nscan,npointing),int)

    ntime=len(chime_bbdata["tiedbeam_baseband"][nscan][0])
    for nscan in range(nscan):
        window[nscan]*=ntime  # set to 1000 for smaller test, max 43670
    
    weight=np.ones((1024,1,1,ntime))
    cross=crosscorr_core(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, calc_results=calcresults,DM=0,
                        index_A=1, index_B=0,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight)

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

    assert peaklag_00 == 0
    assert peaklag_11 == 0

    delays, snrs = extract_subframe_delay(cross[:,0,:,:,:,0])
    assert np.isclose(delays[0,0],-0.11257813,rtol=1e-05) #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.11148437,rtol=1e-05) #should be good to sub nanosecond
    assert snrs[0,0]>=53
    assert snrs[1,1]>=49
    

def test_pulsar():
    #tests whether cross correlation of a pulsar yields expected results based on real data
    
    telescopes = [chime,kko]
    chime_file='/home/calvin/public/astro_255670378_multibeam_B2154+40_chime.h5'
    kko_file='/home/calvin/public/astro_255670378_multibeam_B2154+40_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)

    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ra=329.2578219704166
    dec=40.29612961111111

    DM=71.1239013671875

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

    
    peak_bin=37820
    pulse_lim_low=6000
    pulse_lim_high=6000
    t_a=np.ones((1024,nscan,npointing),int)
    t_a=t_a*(peak_bin-pulse_lim_low)
    R=np.ones((1024,nscan,npointing),int)

    window=np.ones((nscan,npointing),int)
    window*=(pulse_lim_high+pulse_lim_low)
    weight=np.ones((1024,1,1,(pulse_lim_high+pulse_lim_low)))
    cross=crosscorr_core(bbdata_a=chime_bbdata, bbdata_b=out_bbdata, t_a=t_a, window=window, R=R, calc_results=calcresults,DM=DM,
                        index_A=1, index_B=0,sample_rate=2.56,max_lag=max_lag,n_pol=2,
                        weight=weight,fast=False)

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

    assert peaklag_00 == 0
    assert peaklag_11 == 0

    delays, snrs = extract_subframe_delay(cross[:,0,:,:,:,0])
    assert np.isclose(delays[0,0],-0.11945313,rtol=1e-05) #should be good to sub nanosecond
    assert np.isclose(delays[1,1],-0.11828125,rtol=1e-05) #should be good to sub nanosecond
    assert snrs[0,0]>=46
    assert snrs[1,1]>=46

