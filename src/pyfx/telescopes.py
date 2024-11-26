from datetime import datetime

import astropy.units as un
import numpy as np
from astropy import coordinates as ac
from caput.time import Observer

TEL_DTYPE = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("name", "S10"), ("index", "i4")]

# For CHIME position, see https://bao.chimenet.ca/doc/documents/1327
# For Outrigger position, see https://bao.chimenet.ca/doc/documents/1727

chime = ac.EarthLocation.from_geocentric(
    x=-2059166.313 * un.m, y=-3621302.972 * un.m, z=4814304.113 * un.m
)
chime.info.name = "chime"

CHIMELATITUDE = 49.3207092194
CHIMELONGITUDE = -119.6236774310
CHIMEALTITUDE = 555.372

chime_obs = Observer(
    lon=CHIMELONGITUDE,
    lat=CHIMELATITUDE,
    alt=CHIMEALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

#### NEW POSITION, as of July '23 ######
kko = ac.EarthLocation.from_geocentric(
    x=(-2111738.254 - 13.195) * un.m,
    y=(-3581458.222 + 4.144) * un.m,
    z=(4821611.987 - 1.502) * un.m,
)
kko.info.name = "kko"

# Converted geocentric coordinates above using https://www.ngs.noaa.gov/NCAT/
KKOLATITUDE = 49.4189748327
KKOLONGITUDE = -120.5250986421
KKOALTITUDE = 804.497

kko_obs = Observer(
    lon=KKOLONGITUDE,
    lat=KKOLATITUDE,
    alt=KKOALTITUDE,
    lsd_start=datetime(2013, 11, 15),
)

chime = ac.EarthLocation.from_geocentric(
    x=-2059166.313 * un.m, y=-3621302.972 * un.m, z=4814304.113 * un.m
)  # delete me
chime.info.name = "chime"

kko = ac.EarthLocation.from_geocentric(
    x=(-2111738.254 - 10.283) * un.m,
    y=(-3581458.222 + 4.515) * un.m,
    z=(4821611.987 - 3.078) * un.m,
)  # delete me
kko.info.name = "kko"

gbo = ac.EarthLocation.from_geocentric(
    x=883737.348 * un.m, y=-4924481.208 * un.m, z=3943967.891 * un.m
)
gbo.info.name = "gbo"

hco = ac.EarthLocation.from_geocentric(x=0 * un.m, y=0 * un.m, z=0 * un.m)
hco.info.name = "hco"

# TESTBEDS BELOW Calvin calculated TONE Zero from positions provided by Pranav
# https://docs.google.com/spreadsheets/d/1xYgOHYzc09PaL5y0QWJNTREnk-HeyGTegbFXt8zlojo/edit#gid=0

tone = ac.EarthLocation.from_geocentric(
    x=882176.762 * un.m, y=-4925204.140 * un.m, z=3943387.544 * un.m
)
tone.info.name = "tone"

algon = ac.EarthLocation.from_geocentric(
    x=918239.853 * un.m, y=-4346109.580 * un.m, z=4562002.274 * un.m
)
algon.info.name = "algon"


def tels_to_ndarray(tels_list):
    """Convert a (list of) Astropy.EarthLocation object, encoding a telescope, to a Numpy np.ndarray of dtype TEL_DTYPE.
    This is necessary to save the telescope objects to HDF5 attributes.
    """
    if type(tels_list) not in [
        list,
        np.ndarray,
    ]:  # unfortunately, checking __iter__ and __len__ do not work
        tels_list = [tels_list]
    tels_array = np.empty(len(tels_list), dtype=TEL_DTYPE)

    for iitel, tel in enumerate(tels_list):
        tels_array["x"][iitel] = tel.x.to(un.m).value
        tels_array["y"][iitel] = tel.y.to(un.m).value
        tels_array["z"][iitel] = tel.z.to(un.m).value
        tels_array["name"][iitel] = tel.info.name
        tels_array["index"][iitel] = iitel
    return tels_array


def tels_to_astropy(tels_array):
    """Convert an array of dtype TEL_DTYPE toa (list of) telescope objects."""
    tels_list = []
    if type(tels_array) not in [
        list,
        np.ndarray,
    ]:  # unfortunately, checking __iter__ and __len__ do not work
        tels_array = [tels_array]
    for entry in tels_array:
        tel = ac.EarthLocation.from_geocentric(
            x=entry["x"] * un.m, y=entry["y"] * un.m, z=entry["z"] * un.m
        )
        tel.info.name = entry["name"].decode("utf-8")
        tels_list.append(tel)
    if len(tels_list) == 1:
        return tels_list[0]
    else:
        return tels_list


def tel_from_name(name):
    tels = {
        "chime": chime,
        "algon": algon,
        "tone": tone,
        "kko": kko,
        "gbo": gbo,
        "hco": hco,
    }
    return tels[name]
