def test_import():
    from coda.core import VLBIVis


def test_compat():
    import numpy as np
    from coda.core import VLBIVis

    a = VLBIVis()
    dtype = a._dataset_dtypes["time"]
    int_time = np.empty((1024, 10), dtype=dtype)
    assert ("dur_ratio", "<f8") in dtype
    assert ("on_window", bool) in dtype
