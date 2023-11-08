"""Tools for extracting good values of t,w,r for correlator gating.

The main functions provided are get_twr_continuum and get_twr_singlepulse"""

import numpy as np 

def first_valid(arr, axis, invalid_val=0):
    """Gets first valid element as a function of frequency axis.
    arr : np.array 2d
        Pass in np.isfinite(bbdata['tiedbeam_baseband'] as arr
    
    axis : int
        To look for first nonzero element along e.g. the last axis pass in axis = -1.
    
    invalid_val : int
        If no nonzero (i.e. invalid) elements,
    """
    mask = np.isfinite(arr).astype(bool)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def duration(arr,axis = -1):
    """Gets number of valid samples counted over the specified axis."""
    return np.sum(np.isfinite(arr),axis = axis)

def tw2twr(t,w):
    """Converts from freq dependent w to freq dependent r. 
    Here t is assumed to be an index; therefore it is quantized to integers.
    Up to rounding errors in w, this is the inverse of tw2twr()."""
    r_new = w / np.max(w) # one per frequency
    w_new = np.max(w) # one per band
    return t.astype(int), w_new.astype(int), r_new

def twr2tw(t,w,r):
    """Converts from freq dependent r to freq dependent w. 
    Here t is assumed to be an index; therefore it is quantized to intege
    Up to rounding errors in w, this is the inverse of tw2twr().
    """
    return t.astype(int), (w * r).astype(int)

def get_twr_continuum(telA_bbdata,equal_duration = True, pad = 2000):
    """Return t,w,r by looking at the nan pattern in telA_bbdata, making use of ~all the data we have.
    
    This is an appropriate way to get the t,w,r data for a single phase-center pointing on a continuum source.

    To take into account that the different boundaries of the data, we trim the edges of the scan.
    We remove :pad: frames from both the left and right of the integration.t_A & ww.

    Inputs
    ------
    telA_bbdata : BBData
    
    equal_duration : bool
        If equal_duration is set to True, we will make sure the integration time is the same across all frequencies.
    
    min_valid_bw : 0 < float < 1
        Get this fraction of the total bandwidth.
    """
    w = telA_bbdata['tiedbeam_baseband'][:,0] # both polarizations should have same nan pattern because Kotekan works in freq  time element order.
    assert w.shape[0] == 1024
    t_a =  first_valid(w,axis = -1)
    window = duration(w,axis = -1)
    if w.shape[-1] // 2 > np.median(window):
        print(f'WARNING: More than half of the channels invalid in data for {telA_bbdata.attrs["event_id"]}')
    if equal_duration: # choose the length that maximizes the sensitivity as quantified by num_valid_channels x duration
        sorted_window_lengths = np.sort(window)
        sens_metric = sorted_window_lengths * (1024 - np.arange(1024)) # n_time * n_channels with that many valid samples
        window = sorted_window_lengths[np.argmax(sens_metric)] + np.zeros(1024,dtype=int)
    assert (np.min(window) - 2 * pad > 0), "twr params result in negative integration duration when zero-padded. Please optimize manually, e.g. decrease pad value or input twr manually."
    tt,ww,rr =tw2twr(t_a + pad, window - 2 * pad)
    ww = np.atleast_2d(ww)
    tt.shape = (1024,1,1)
    rr.shape = (1024,1,1)
    return tt,ww,rr
