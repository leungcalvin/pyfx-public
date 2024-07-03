import numpy as np
from pyfx.core_correlation import getitem_zp1d

def test_getitem_zp1d_basic():
    arr = np.arange(100) + 1000j

    # no data
    assert getitem_zp1d(arr,0,0).size == 0, "Should have size zero"

    # single data point
    assert getitem_zp1d(arr,1,2) == arr[1:2]

    # request all in data
    ans = np.array([19,20,21,22,23,24]) + 1000j
    assert np.allclose(getitem_zp1d(arr,19,25),ans)
    
    # data all within request
    ans = arr
    assert np.allclose(getitem_zp1d(arr,-2,102)[2:102],ans)
    assert np.isclose(getitem_zp1d(arr,-2,102)[102:],0).all()
    assert np.isclose(getitem_zp1d(arr,-2,102)[:2],0).all()

def test_getitem_zp1d_advanced():
    arr = np.arange(100) + 1000j
    
    # data early
    ans = np.array([99 + 1000j,0,0,0,0,0])
    assert np.allclose(getitem_zp1d(arr,99,105),ans)

    # data late
    ans = np.array([0,0+ 1000j,1+ 1000j,2+ 1000j,3+ 1000j,4+ 1000j]) 
    assert np.allclose(getitem_zp1d(arr, -1,5),ans)

    # data out 
    ans = np.array([0]) 
    assert np.allclose(getitem_zp1d(arr, -1,0),ans)

    # data out
    assert np.allclose(getitem_zp1d(arr, -99,-10),np.zeros(89,dtype = complex))
    
    # data out
    assert np.allclose(getitem_zp1d(arr, 200,210),np.zeros(10,dtype = complex))

def test_ctime2atime():
    from astropy.time import Time
    from pyfx.corr_job_station import ctimeo2atime, atime2ctimeo
    aa =Time(val = 1719999940, val2 = 0.1234567890,format='unix')
    bb = ctimeo2atime(*atime2ctimeo(ctimeo2atime(*atime2ctimeo(aa))))
    assert np.isclose(bb.to_value('unix','long'),aa.to_value('unix','long'))
    assert np.isclose(bb.to_value('unix','float'),aa.to_value('unix','float'))
