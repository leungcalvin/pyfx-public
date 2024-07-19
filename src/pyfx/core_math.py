"""A module which holds bindings to mathematical operations. This module is meant to be used on *baseband* data"""
from baseband_analysis.core import BBData
import numpy as np

def xy_to_circ(bb):
    assert bb.shape[1] == 2, "Baseband is not dual-polarization!"
    if bb.ndim != 3:
        UserWarning("Data shape is {bb.shape}! Expected 3 dimensions.")
    out = np.zeros_like(bb)
    out[:, 0] = bb[:, 0] + 1j * bb[:, 1]
    out[:, 1] = bb[:, 0] - 1j * bb[:, 1]
    return out

def get_kurtosis_estimator(
    baseband_data:np.ndarray, #(nfreq,npol,ntime)
    ):
    """
    see Gelu M. Nita 2010: https://web.njit.edu/~gary/assets/Nita_RFI2010.pdf
    """
    mom_2=np.nansum(np.abs(baseband_data)**2,axis=-1) #nfreq,npol
    mom_4=np.nansum(np.abs(baseband_data)**4,axis=-1) #nfreq,npol
    N_vals=[len(np.where(baseband_data[i,0,:]>0.0)[0]) for i in range(len(baseband_data))]
    N_vals=np.array(N_vals)
    scale=(N_vals+1)/(N_vals-1)
    return  scale[:,np.newaxis]*(N_vals[:,np.newaxis]*mom_4/mom_2**2-1)#nfreq,npol

def get_sk_rfi_mask(
    bbdata:BBData
    ):
    #will need to save this to VLBIVis file; e.g: vis['chime'].create_dataset('sk', data=sk_values)
    return get_kurtosis_estimator(bbdata['tiedbeam_baseband'])
    
