import numpy as np
import astropy.coordinates as ac
import astropy.units as un
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import fill_waterfall
from astropy.time import Time,TimeDelta
import scipy
from pyfx import corr_job_station as corr_job
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
    from coda.core import VLBIVis
    pointing_spec = np.empty((1,),dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = ra
    pointing_spec['corr_dec'][:] = dec
    pointing_spec['source_name'][:] = 'NCP_pytest'
    pointing_spec['dm_correlator'][:] = 0

    ncp_job = corr_job.CorrJob(
        bbdatas=[chime_bbdata,out_bbdata],
        telescopes=telescopes,
        pointing_spec = pointing_spec,
        ref_station = 'chime',
        )

    gate_spec = ncp_job.define_scan_params_transient(
        start_or_toa = 'start',
        t0f0 = ('start','top'),
        time_spacing = 'even',
        freq_offset_mode = 'bbdata',
        window = chime_bbdata.ntime // 10,
        r_ij = np.ones(1024),
        period_frames = chime_bbdata.ntime // 10,
        num_scans_before = 10,
        num_scans_after = 8,
    )
    nstation, nfreq, npointing, ntime = (2, 1024, 1, 10 + 8 + 1)
    assert gate_spec.shape == (nfreq, npointing, ntime)
    assert (np.abs(gate_spec['gate_start_unix'][:,0,0] - chime_bbdata['time0']['ctime'][:]) < FLOAT64_PRECISION).all(), "Expected start times to start at the BBData edge for the reference station"

def test_corr_job_runs_no_fill():
    """Same as the above, but no fill_waterfall"""
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)
    ra=np.atleast_1d(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.atleast_1d(chime_bbdata['tiedbeam_locations']['dec'][0])
    telescopes = [chime,kko]
    #fill_waterfall(chime_bbdata, write=True)
    #fill_waterfall(out_bbdata, write=True)
    from coda.core import VLBIVis
    pointing_spec = np.empty((1,),dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = ra
    pointing_spec['corr_dec'][:] = dec
    pointing_spec['source_name'][:] = 'NCP_pytest'
    pointing_spec['dm_correlator'][:] = 0

    ncp_job = corr_job.CorrJob(
        bbdatas=[chime_bbdata,out_bbdata],
        telescopes=telescopes,
        pointing_spec = pointing_spec,
        ref_station = 'chime',
        )

    gate_spec = ncp_job.define_scan_params_transient(
        start_or_toa = 'start',
        t0f0 = ('start','top'),
        time_spacing = 'even',
        freq_offset_mode = 'bbdata',
        window = chime_bbdata.ntime // 10,
        r_ij = np.ones(1024),
        period_frames = chime_bbdata.ntime // 10,
        num_scans_before = 10,
        num_scans_after = 8,
    )
    nstation, nfreq, npointing, ntime = (2, 1024, 1, 10 + 8 + 1)
    assert gate_spec.shape == (nfreq, npointing, ntime)
    assert (np.abs(gate_spec['gate_start_unix'][:,0,0] - chime_bbdata['time0']['ctime'][:]) < FLOAT64_PRECISION).all(), "Expected start times to start at the BBData edge for the reference station"

def test_continuum_calibrator_corrjob():
    # note: dry_atm matters in delay model, values below are for dry_atm=True,wet_atm=True
    telescopes = [chime,kko]
    chime_file='/arc/projects/chime_frb/pyfx_test_files/J0117+8928_chime.h5' 
    kko_file='/arc/projects/chime_frb/pyfx_test_files/J0117+8928_kko.h5'
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)

    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)

    ras = np.array([chime_bbdata["tiedbeam_locations"]["ra"][0]])
    decs = np.array([chime_bbdata["tiedbeam_locations"]["dec"][0]])
    source_names = np.array(['J0117+8928'])
    bbatas=[chime_bbdata,out_bbdata]

    # Just one pointing
    from coda.core import VLBIVis
    pointing_spec = np.empty((1,),dtype = VLBIVis._dataset_dtypes['pointing'])
    pointing_spec['corr_ra'][:] = ras
    pointing_spec['corr_dec'][:] = decs
    pointing_spec['source_name'][:] = 'J0117+8928_pytest'
    pointing_spec['dm_correlator'][:] = 0
    ss_job = corr_job.CorrJob(bbdatas=bbatas,
    telescopes=telescopes, pointing_spec = pointing_spec)
    gate_spec = ss_job.define_scan_params_continuum(pad = 0)
    #from pyfx.twr_utils import get_tw_frame_continuum
    #tt,ww,rr=get_tw_frame_continuum(ss_job.bbdatas[0],pad=0)
    #tt_station=[tt,tt] # ¯\_(ツ)_/¯ only matters for autos

    vis = ss_job.run_correlator_job(event_id = 304295669,gate_spec=gate_spec,pointing_spec=pointing_spec,out_h5_file = False)

    cross=vis['chime-kko']['vis'][:]

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

