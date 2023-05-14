import pytest
import numpy as np
from pyfx.core_correlation import intrachannel_dedisp,frac_samp_shift,autocorr_core
from pyfx.core_correlation_vectorized import intrachannel_dedisp_vectorized,frac_samp_shift_vectorized,autocorr_core_vectorized

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

def test_dedispersion_vectorized():
    #consistency check between vectorized and non-vectorized versions
    ntime=7
    nfreq=9
    DM=np.random.uniform(30,500)
    data=np.random.uniform(-5e6,5e6,(nfreq,8,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,8,ntime))
    f0=np.linspace(400,800,9)
    vectorized_dedispersed=intrachannel_dedisp_vectorized(data,DM,f0)
    for freq in range(len(f0)):
        dedispersed=intrachannel_dedisp(data[freq],DM,f0[freq])
        np.testing.assert_array_equal(vectorized_dedispersed[freq],dedispersed)

def test_fracsampleshift_vectorized():
    #consistency check between vectorized and non-vectorized versions
    ntime=7
    nfreq=9
    data=np.random.uniform(-5e6,5e6,(nfreq,8,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,8,ntime))
    f0=np.linspace(400,800,9)
    sub_frame_tau=np.random.uniform(-1,1,(nfreq,ntime))
    vectorized_frac_sample=frac_samp_shift_vectorized(data, f0, sub_frame_tau)
    for freq in range(len(f0)):
        frac_sample=frac_samp_shift(data[freq],f0[freq],sub_frame_tau[freq])
        np.testing.assert_array_equal(vectorized_frac_sample[freq],frac_sample)

'''def test_autocorr_vectorized_1():
    #consistency check between vectorized and non-vectorized versions
    ntime=100
    nfreq=9
    f0=np.linspace(400,800,nfreq)
    npol=2
    nscan=1
    t_a=np.random.randint(0,10,size=(nfreq,nscan))*0+10
    data=np.random.uniform(-5e6,5e6,(nfreq,npol*nscan,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,npol*nscan,ntime))
    bbdata_a=VeryBasicBBData(nfreq=f0,data=data)
    R=np.random.uniform(.5,1,(nfreq,nscan))
    max_lag=10
    window=np.ones(nscan)*40
    vectorized_autocorr=autocorr_core_vectorized(bbdata_a=bbdata_a, t_a=t_a,window=window,R=R,max_lag=max_lag, DM = 0)
    window=np.ones((nfreq,nscan))*40
    autocorr=autocorr_core(bbdata_A=bbdata_a, T_A=t_a,Window=window,R=R,max_lag=max_lag,DM = 0)
    np.testing.assert_array_equal(vectorized_autocorr,autocorr)
    for i in range(nscan):
        assert (np.imag(vectorized_autocorr[...,i,0,0,0,0])==0.0).all()==True
        assert (np.imag(vectorized_autocorr[...,i,1,1,0,0])==0.0).all()==True
'''

def test_autocorr():
    #tests whether output of autocorr makes sense given input data
    ntime=100
    nfreq = 9
    freq_ids_present =np.arange(nfreq)
    npol=2
    nscan=1
    t_a=np.random.randint(0,10,size=(nfreq,nscan))*0+10
    data=(np.random.uniform(-5e6,5e6,(nfreq,npol*nscan,ntime))+1j*np.random.uniform(-5e6,5e6,(nfreq,npol*nscan,ntime)))*0
    amplitude_00=8+0j
    amplitude_11=50+0j
    for iifreq in range(nfreq):
        for scan in range(nscan):
            data[iifreq,scan*2,t_a[iifreq,scan]+1]=2+2j #peakval 8
            data[iifreq,scan*2+1,t_a[iifreq,scan]+1]=5+5j #peakval 50
            
    bbdata_a=VeryBasicBBData(freq_ids=freq_ids_present,data=data)
    R=np.ones((nfreq,nscan))
    max_lag=10
    window=np.ones(nscan)*40
    vectorized_autocorr=autocorr_core_vectorized(DM = 1.2,bbdata_a=bbdata_a, t_a=t_a,window=window,R=R,max_lag=max_lag)
    window=np.ones((nfreq,nscan))*40
    assert np.isclose(vectorized_autocorr[...,:,0,0,0,0],amplitude_00).all()
    assert np.isclose(vectorized_autocorr[...,:,1,1,0,0],amplitude_11).all()
