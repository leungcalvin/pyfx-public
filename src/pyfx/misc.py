"""Miscellaneous utilities for PyFX."""
from difxcalc_wrapper import telescopes

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
            if key in bbdata.attrs['gain_cal_h5']:
                return names[key]
    return 'algon' # because this is the weird child
