import numpy as np
import astropy.coordinates as ac
import astropy.units as un
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import fill_waterfall
from astropy.time import Time,TimeDelta
import scipy
from pyfx import corr_job
from pyfx import corr_job_station 
from pycalc11 import Calc
from pyfx.core_vis import extract_subframe_delay, extract_frame_delay
import logging
chime_file='/arc/projects/chime_frb/pyfx_test_files/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_chime.h5'
kko_file='/arc/projects/chime_frb/pyfx_test_files/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_kko.h5'
FLOAT64_PRECISION = 2 * 2**-22 #our times should be this good: https://www.leebutterman.com/2021/02/01/store-your-unix-epoch-times-as-float64.html

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

def test_corr_job_runs_filled():
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    ra=np.atleast_1d(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.atleast_1d(chime_bbdata['tiedbeam_locations']['dec'][0])
    telescopes = [chime,kko]
    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)
    ncp_job = corr_job.CorrJob([chime_file,kko_file],telescopes=telescopes,
        ras = ra,
        decs = dec,
        source_names=np.atleast_1d(["NCP"])
        )

    t,w,r = ncp_job.define_scan_params(
        ref_station = 'chime',
        start_or_toa = 'start',
        t0f0 = ('start','top'),
        time_spacing = 'even',
        freq_offset_mode = 'bbdata',
        Window = np.ones(1024) * chime_bbdata.ntime // 10,
        r_ij = np.ones(1024),
        period_frames = chime_bbdata.ntime // 10,
        num_scans_before = 10,
        num_scans_after = 8,
    )
    nstation, nfreq, npointing, ntime = (2, 1024, 1, 10 + 8 + 1)
    assert t.shape == (nstation, nfreq, npointing, ntime)
    assert (np.abs(t[0,:,0,0] - chime_bbdata['time0']['ctime'][:]) < FLOAT64_PRECISION).all(), "Expected start times to start at the BBData edge for the reference station"
    assert w.shape == (nfreq, npointing, ntime)
    assert r.shape == (nfreq, npointing, ntime)

def test_corr_job_runs_no_fill():
    """Same as the above, but no fill_waterfall"""
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    ra=np.atleast_1d(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.atleast_1d(chime_bbdata['tiedbeam_locations']['dec'][0])
    telescopes = [chime,kko]
    ncp_job = corr_job.CorrJob([chime_file,kko_file],telescopes=telescopes,
        ras = ra,
        decs = dec,
        source_names=np.atleast_1d(["NCP"])
        )
    t,w,r = ncp_job.define_scan_params(
        ref_station = 'chime',
        start_or_toa = 'start',
        t0f0 = ('start','top'),
        time_spacing = 'even',
        freq_offset_mode = 'bbdata',
        Window = np.ones(1024) * chime_bbdata.ntime // 10,
        r_ij = np.ones(1024),
        period_frames = chime_bbdata.ntime // 10,
        num_scans_before = 10,
        num_scans_after = 8,
    )
    nstation, nfreq, npointing, ntime = (2, 1024, 1, 10 + 8 + 1)
    freqs_present_chime = chime_bbdata.index_map['freq']['id'][:]
    freqs_present_kko = out_bbdata.index_map['freq']['id'][:]
    assert t.shape == (nstation, nfreq, npointing, ntime)
    assert (np.abs(t[0,freqs_present_chime,0,0] - chime_bbdata['time0']['ctime'][:]) < FLOAT64_PRECISION).all(), "Expected start times to start at the BBData edge for the reference station"
    assert w.shape == (nfreq, npointing, ntime)
    assert r.shape == (nfreq, npointing, ntime)


def test_pulsar_pycalc_corrjob():
    """Tests whether cross correlation of a pulsar yields expected results based on real data using pycalc in crosscorr_core. 
    Same as test_pulsar_pycalc() but using pycalc instead of difxcalc-wrapper in CorrJob
    Run this on CANFAR in a container containing pycalc, pyfx, and baseband-analysis.
    """
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
    pulsar_job = corr_job.CorrJob([chime_file,kko_file],telescopes=telescopes,
       ras = ra,
       decs = dec,
       source_names=np.atleast_1d('B0355+54')
	   )
    toa=chime_bbdata['time0']['ctime'][0]+25310*2.56e-6
    t,w,r = pulsar_job.define_scan_params(ref_station = 'chime',
			      start_or_toa = 'start',
			      t0f0 = (toa, 800.0),#(1689783027.6518016, 800.0),
			      freq_offset_mode = 'dm',
			      Window = np.ones(1024) * 761,
			      r_ij = np.ones(1024) * 1,
			      dm = 57.1,
                  max_lag=100,
			      )

    vlbivis = pulsar_job.run_correlator_job(t[...,0:1],w[0,:,0:1].astype(int),r[...,0:1],dm = 57.1,
			      out_h5_file = False)
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
    
    assert snrs_pycalc[0,0]>=30, f"fringe signal to noise is below expected value in 0,0 pol, got {snrs_pycalc[0,0]}"
    assert snrs_pycalc[1,1]>=27, f"fringe signal to noise is below expected value in 1,1 pol, got {snrs_pycalc[1,1]}"
