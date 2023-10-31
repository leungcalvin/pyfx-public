"""Utilities for reading BBData without loading in the full dataset."""

import h5py

def station_from_bbdata(bbdata):
    """Returns station name for use with difxcalc"""
    method = 'index_map'
    if method == 'index_map':
        first_three_inputs = [s.decode('utf-8')[0:3] for s in bbdata.index_map['input']['correlator_input'][0:3]]
        prefixes = {'FCC':'chime',
                    'FCA':'kko',
                    'FCB':'gbo',
                    'TON':'tone'}
        for key in prefixes.keys():
            if key in first_three_inputs:
                return prefixes[key]
    # if we have not found anything so far, try another method
    method = 'gains'
    if method == 'gains':
        names = {'kko':'kko',
                 'pco':'kko',
                 'chime':'chime',
                 'tone':'tone',
                 'gbo':'gbo'}
        for key in names.keys():
            if key in bbdata.attrs['cal_h5']:
                return names[key]
    return 'algon' # because this is the weird child

def get_all_time0(bbdata_filename, method = 'single'):
    """Bypass caput slicing to read data["time0"]"""
    if method == 'single': # only handle all frequencies in a single file for now. TODO: do this with a caput Reader.
        with h5py.File(bbdata_filename, mode = 'r') as f:
            return f['time0'][:].copy()
    if method == 'multiple':
        raise NotImplementedError('Should probably grep the filepath for something like baseband_XXX_FREQID and read them in a loop')

def get_all_im_freq(bbdata_filename, method = 'single'):
    """Bypass caput slicing to read data.index_map['freq']"""
    if method == 'single': # only handle all frequencies in a single file for now. TODO: do this with a caput Reader.
        with h5py.File(bbdata_filename,mode = 'r') as f:
            return f['index_map/freq'][:].copy()
    if method == 'multiple':
        raise NotImplementedError('Should probably grep the filepath for something like baseband_XXX_FREQID and read them in a loop')

def choose_beam_idx_from_pointing(pointing_ra, pointing_dec, tiedbeam_ra, tiedbeam_dec,tolerance_deg = 2/60):
    assert type(pointing_ra) is float, "One pointing a time!"
    assert type(pointing_dec) is float, "One pointing at a time!"
    pairwise_distances_deg = ((pointing_ra - tiedbeam_ra)**2 + (pointing_dec - tiedbeam_dec)**2 * np.cos(pointing_dec * np.pi / 180) **2 )**0.5
    assert np.min(pairwise_distances_deg) < tolerance_deg, "No suitable beam found for correlator pointing."
    return np.argmin(pairwise_distances_deg)

def get_ntime(bbdata_filename):
    with h5py.File(bbdata_filename, mode = 'r') as f:
        return f['tiedbeam_baseband'].shape[-1]
    return 
