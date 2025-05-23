"""Utilities for reading BBData without loading in the full dataset."""

from glob import glob
from typing import List
import logging
import h5py
import numpy as np
from baseband_analysis.core import BBData
from baseband_analysis.core.bbdata import concatenate as concat_bbdatas
from baseband_analysis.core.sampling import fill_waterfall


def station_from_bbdata(bbdata, method="index_map"):
    """Returns station name for use with difxcalc"""
    if method == "index_map":
        first_three_inputs = [
            s.decode("utf-8")[0:3]
            for s in bbdata.index_map["input"]["correlator_input"][0:3]
        ]
        prefixes = {"FCC": "chime", "FCA": "kko", "FCB": "gbo", "TON": "tone"}
        for key in prefixes.keys():
            if key in first_three_inputs:
                return prefixes[key]
    # if we have not found anything so far, try another method
    method = "gains"
    if method == "gains":
        names = {
            "kko": "kko",
            "pco": "kko",
            "tone": "tone",
            "gbo": "gbo",
            "chime": "chime",
        }
        for key in names.keys():
            if key in bbdata.attrs["cal_h5"]:
                return names[key]
    return "algon"  # because this is the weird child


def get_all_time0(bbdata_filename, method="single"):
    """Bypass caput slicing to read data["time0"]"""
    if (
        method == "single"
    ):  # only handle all frequencies in a single file for now. TODO: do this with a caput Reader.
        with h5py.File(bbdata_filename, mode="r") as f:
            return f["time0"][:].copy()
    if method == "multiple":
        raise NotImplementedError(
            "Should probably grep the filepath for something like baseband_XXX_FREQID and read them in a loop"
        )


def get_all_im_freq(bbdata_filename, method="single"):
    """Bypass caput slicing to read data.index_map['freq']"""
    if (
        method == "single"
    ):  # only handle all frequencies in a single file for now. TODO: do this with a caput Reader.
        with h5py.File(bbdata_filename, mode="r") as f:
            return f["index_map/freq"][:].copy()
    if method == "multiple":
        raise NotImplementedError(
            "Should probably grep the filepath for something like baseband_XXX_FREQID and read them in a loop"
        )


def get_multibeam_pointing(bbdata_multibeam_filename, method="single"):
    """Bypass caput slicing to read data.index_map['freq']"""
    if (
        method == "single"
    ):  # only handle all frequencies in a single file for now. TODO: do this with a caput Reader.
        with h5py.File(bbdata_multibeam_filename, mode="r") as f:
            return f["tiedbeam_locations"][:].copy()  # (ra,dec,...)
    if method == "multiple":
        raise NotImplementedError(
            "Should probably grep the filepath for something like baseband_XXX_FREQID and read them in a loop"
        )


def choose_beam_idx_from_pointing(
    pointing_ra, pointing_dec, tiedbeam_ra, tiedbeam_dec, tolerance_deg=2 / 60
):
    assert type(pointing_ra) is float, "One pointing a time!"
    assert type(pointing_dec) is float, "One pointing at a time!"
    pairwise_distances_deg = (
        (pointing_ra - tiedbeam_ra) ** 2
        + (pointing_dec - tiedbeam_dec) ** 2 * np.cos(pointing_dec * np.pi / 180) ** 2
    ) ** 0.5
    assert (
        np.min(pairwise_distances_deg) < tolerance_deg
    ), "No suitable beam found for correlator pointing."
    return np.argmin(pairwise_distances_deg)


def get_ntime(bbdata_filename):
    with h5py.File(bbdata_filename, mode="r") as f:
        return f["tiedbeam_baseband"].shape[-1]
    return


### multibeam singlebeam crap


def get_bbdatas_from_index(
    tel_beamformed_dirs: List[str],
    tel_a_n: int,
    ref_index: int = 0,  # index of telescope bbdata dictating pointing
    fill_freqs: bool = False,
):
    ## assuming 0th index is telescope A, although this doesn't really matter much
    tel_bbdatas = []

    bbdata_a = extract_singlebeam(
        tel_beamformed_dir=tel_beamformed_dirs[ref_index], n=tel_a_n
    )

    ra = bbdata_a["tiedbeam_locations"]["ra"][0]
    dec = bbdata_a["tiedbeam_locations"]["dec"][0]
    tel_beamformed_dirs.pop(ref_index)
    tel_bbdatas = get_bbdatas_from_pointing(
        tel_beamformed_dirs=tel_beamformed_dirs, ra=ra, dec=dec, fill_freqs=fill_freqs
    )
    if fill_freqs:
        fill_waterfall(bbdata_a, write=True)
    tel_bbdatas.insert(0, bbdata_a)  # insert bbdata_a to 0th index
    return tel_bbdatas


def get_bbdatas_from_pointing(
    tel_beamformed_dirs: List[str],
    ra: float,
    dec: float,
    fill_freqs: bool,
):
    ## assuming 0th index is telescope A, although this doesn't really matter much
    tel_bbdatas = []
    for tel_beamformed_dir in tel_beamformed_dirs:
        tel_b_file = glob(tel_beamformed_dir)[0]
        pointings_b = get_multibeam_pointing(tel_b_file)
        tel_b_n = get_pointing_index(ra=ra, dec=dec, pointings_tel=pointings_b)
        assert (
            len(tel_b_n) > 0
        ), f"no tiedbeam pointing matching telescope a beamformed data from data in {tel_beamformed_dir}"
        bbdata_b = extract_singlebeam(
            tel_beamformed_dir=tel_beamformed_dir, n=tel_b_n[0]
        )
        if fill_freqs:
            fill_waterfall(bbdata_b, write=True)
        tel_bbdatas.append(bbdata_b)
    return tel_bbdatas


def get_pointing_index(ra: float, dec: float, pointings_tel: np.ndarray):
    """
    Get index/indices of multibeam data corresponding to specified ra and dec
    """
    dec = pointings_tel["dec"][0]
    ra = pointings_tel["ra"][0]
    dec_matches = np.isclose(
        pointings_tel["dec"], dec, rtol=0.1 / 3600
    )  # should be the same to 100mas"
    ra_matches = np.isclose(
        pointings_tel["ra"], ra, rtol=0.1 / (3600 * np.cos(np.deg2rad(dec)))
    )  # should be the same to 100mas"
    indices = np.where((ra_matches) & (dec_matches))[0]
    return indices


def extract_singlebeam(tel_beamformed_dir: str, n: int) -> BBData:
    """
    Extracts the nth source in the multibeam data and consolidates it into a singlebeam bbdata object.

    Inputs:
    ----------
    tel_beamformed_dir: str
        directory where multibeam data is stored
    n: int
        integer denoting the nth source in the beamformed data that will be extracted and consolidated

    Returns:
    ----------
    BBData object containing singlebeam data for the specified nth source
    """

    if tel_beamformed_dir[-1] == "*":
        tel_beamformed_dir += ".h5"
    elif tel_beamformed_dir[-1] == "/":
        tel_beamformed_dir += "*.h5"
    elif tel_beamformed_dir[-1] != "5":
        tel_beamformed_dir += "/*.h5"

    tel_files = glob(tel_beamformed_dir)
    datas = []
    assert len(tel_files) > 0, f"no files found in glob({tel_beamformed_dir})"
    for file in tel_files:
        try:
            datas.append(BBData.from_file(file, beam_sel=slice(2 * n, 2 * n + 2, 1)))
        except:
            logging.info(f"could not read in {file}")
    tel_bbdata = concat_bbdatas(datas)
    tel_bbdata_freqs = tel_bbdata.index_map["freq"]["centre"]

    assert len(np.unique(tel_bbdata_freqs)) == len(
        tel_bbdata_freqs
    ), f"a redundent number of frequencies exist in {tel_beamformed_dir}. Please remove redundant frequency files before proceeding"
    return tel_bbdata
