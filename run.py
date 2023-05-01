import importlib
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import os
import astropy.units as un
import astropy.coordinates as ac
from astropy.time import Time
import copy

import difxcalc_wrapper.runner as dcr
from difxcalc_wrapper.io import make_calc

#private packages
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.sampling import fill_waterfall

DIFXCALC_CMD='/home/shion/projects/rrg-vkaspi-ad/shion/difxcalc11/build/bin/difxcalc'
DIFXCALC_SCRATCH_DIR='/home/shion/projects/rrg-vkaspi-ad/shion/'
from baseband_analysis.dev.vis_io import get_pointing_center

from pyfx.core_correlation_vectorized import crosscorr_core_freq_vectorized
from .core_math import max_lag_slice
from .core_correlation import crosscorr_core
from .core_correlation import autocorr_core

# defining telescope positions explicitly here for convenience 
chime = ac.EarthLocation.from_geocentric(
    x = -2059166.313 * un.m,
    y = -3621302.972 * un.m,
    z =  4814304.113 * un.m)
chime.info.name = 'chime'

kko = ac.EarthLocation.from_geocentric(
    x = -2111738.254 * un.m, 
    y = -3581458.222 * un.m,  
    z = 4821611.987 * un.m) 
kko.info.name = 'kko'

telescopes = [chime,kko] 




chime_file='/home/shion/projects/rrg-vkaspi-ad/shion/singlebeam_255860695_NCP_kko.h5'
kko_file='/home/shion/projects/rrg-vkaspi-ad/shion/singlebeam_255860695_NCP_kko.h5'
chime_bbdata = BBData.from_file(chime_file)
out_bbdata = BBData.from_file(kko_file)

fill_waterfall(chime_bbdata, write=True)
fill_waterfall(out_bbdata, write=True)

ra=np.array(chime_bbdata['tiedbeam_locations']['ra'][0])
dec=np.array(chime_bbdata['tiedbeam_locations']['dec'][0])
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
    remove_calcfile=True,
    force=True,
)


    

telA=chime_bbdata
T_A=np.zeros((1024,1),int)
T_A_new=T_A
R=np.ones((1024,1),int)

Window=np.ones((1024,1),int)
for freq in range(len(T_A)):
    Window[freq]*=len(telA["tiedbeam_baseband"][freq][0])
DM=0
max_lag=100
import time

start = time.time()
cross=crosscorr_core(bbdata_A=chime_bbdata, bbdata_B=out_bbdata, T_A=T_A_new, Window=Window, R=R, calc_results=calcresults,DM=DM,
                     index_A=1, index_B=0,sample_rate=2.56,max_lag=max_lag,n_pol=2)
end = time.time()
print(end - start)

telA=chime_bbdata
T_A=np.zeros((1024,1),int)
T_A_new=T_A
R=np.ones((1024,1),int)

Window=np.ones((1024,1),int)
for freq in range(len(T_A)):
    Window[freq]*=len(telA["tiedbeam_baseband"][freq][0])
DM=0
max_lag=100
import time

start = time.time()
cross=crosscorr_core(bbdata_A=chime_bbdata, bbdata_B=out_bbdata, T_A=T_A_new, Window=Window, R=R, calc_results=calcresults,DM=DM,
                     index_A=1, index_B=0,sample_rate=2.56,max_lag=max_lag,n_pol=2)
end = time.time()
print(end - start)

Window=len(telA["tiedbeam_baseband"][freq][0])
start = time.time()
cross=crosscorr_core_freq_vectorized(bbdata_A=chime_bbdata, bbdata_B=out_bbdata, T_A=T_A_new, Window=Window, R=R, calc_results=calcresults,DM=DM,
                     index_A=1, index_B=0,sample_rate=2.56,max_lag=max_lag,n_pol=2)
end = time.time()
print(end - start)