"""Configuration parameters for PyFX which might change for different users.

You might want to configure:

1) The working directory of the delay model,
2) The channelization of your telescope.
"""
import numpy as np

CALCFILE_DIR = '/scratch/calvin' # where .calc files are temporarily saved during CorrJobs

"""Different channelization parameters supported in PyFX. Add yours here!"""
CHIME_PFB = {
    'nchan' : 1024, # Number of frequency channels, not including the Nyquist frequency.
    'lblock' : 2048, # Number of samples in a block. Usu. a fast FFT length; for critically-sampled PFB this is 2*nchan.
    'ntap' : 4, # Number of PFB taps
    'freq_mhz' : np.linspace(800,400,num = 1024, endpoint = False),
    'frame_microseconds' : 2.56, # In microseconds.
    'chan_bw' : 390.625, # Full-width in kHz
    'window_type' : 'sinc_hann', # TODO: Implement options besides 'sinc_hann', currently this field is unused.
    'search_lags' : [   0.        ,  341.33333333,  682.66666667, -1024.0 , -682.66666667, -341.33333333], # Sub-integer lags (should be less than lblock) over which to search.
    'nlags' : 100, # Number of quasi-integer lags to keep. 
}
"""A note about nlags: 
For all of the correlators this is the number of integer lags and the maximum delay range. 
For the search correlator we oversample in lag space by a factor of n_search_lags...
...we keep this many points corresponding to a delay range of `nlags / n_search_lags`."""

CHIME_FFT = {
    'nchan' : 1024,
    'lblock' : 2048,
    'ntap' : 1,
    'fmin' : 400.390625,
    'frame_microseconds' : 2.56,
    'chan_bw' : 390.625, # KHz
    'window_type' : 'rectangle',
    'search_lags' : [0,512,1024],
    'nlags' : 40
}

LITTLE_CRITICAL_PFB = {
    'nchan' : 64,
    'lblock' : 128,
    'ntap' : 4,
    'fmin' : 400.390625,
    'frame_microseconds' : 2.56,
    'chan_bw' : 390.625, # KHz
    'window_type' : 'sinc_hann',
    'search_lags' : [  0.        ,  21.33333333,  42.66666667,  64.        , 85.33333333, 106.66666667, 128.        ],
    'nlags' : 40,    
}

CHANNELIZATION = CHIME_PFB
