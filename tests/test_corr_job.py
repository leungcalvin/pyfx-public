import numpy as np
import astropy.coordinates as ac
import astropy.units as un
from baseband_analysis.core import BBData
from baseband_analysis.core.sampling import fill_waterfall

from pyfx import corr_job

chime_file='/home/calvin/public/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_chime.h5'
kko_file='/home/calvin/public/astro_256150292_multibeam_LOFAR_L725386_24.2440833_47.8580556_kko.h5'
FLOAT64_PRECISION = 2 * 2**-22 #our times should be this good: https://www.leebutterman.com/2021/02/01/store-your-unix-epoch-times-as-float64.html

def test_corr_job_runs_filled():
    chime_bbdata = BBData.from_file(chime_file)
    out_bbdata = BBData.from_file(kko_file)

    fill_waterfall(chime_bbdata, write=True)
    fill_waterfall(out_bbdata, write=True)

    ra=np.array(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.array(chime_bbdata['tiedbeam_locations']['dec'][0])
    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]
    ncp_job = corr_job.CorrJob([chime_file,kko_file],
        ras = np.array(chime_bbdata['tiedbeam_locations']['ra'][0]),
        decs = np.array(chime_bbdata['tiedbeam_locations']['dec'][0])
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

    ra=np.array(chime_bbdata['tiedbeam_locations']['ra'][0])
    dec=np.array(chime_bbdata['tiedbeam_locations']['dec'][0])
    sources = [ac.SkyCoord(ra=ra * un.degree, dec=dec * un.degree, frame="icrs")]
    ncp_job = corr_job.CorrJob([chime_file,kko_file],
        ras = np.array(chime_bbdata['tiedbeam_locations']['ra'][0]),
        decs = np.array(chime_bbdata['tiedbeam_locations']['dec'][0])
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
