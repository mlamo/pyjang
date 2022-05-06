"""Gravitational wave information."""

import astropy.time
from cached_property import cached_property
import h5py
import healpy as hp
from ligo.skymap.io import fits
import logging
import numpy as np
import os
import pandas as pd
from typing import Optional, Tuple

import jang.conversions
import jang.neutrinos
import jang.parameters

priorities = [
    "C01:IMRPhenomXPHM",
    "IMRPhenomXPHM",
    "C01:IMRPhenomPv3HM",
    "IMRPhenomPv3HM",
    "C01:IMRPhenomPv2",
    "IMRPhenomPv2",
    "C01:IMRPhenomNSBH:HighSpin",
    "IMRPhenomNSBH:HighSpin",
    "C01:IMRPhenomNSBH:LowSpin",
    "IMRPhenomNSBH:LowSpin",
    "C01:Mixed",
    "Mixed",
    "PublicationSamples"
]


class GW:
    def __init__(
        self,
        name: str = None,
        path_to_fits: str = None,
        path_to_samples: str = None,
        logger: str = "jang",
    ):
        self.name = name
        self.fits = None
        self.samples = None
        self.logger = logger
        self.catalog = ''
        if path_to_fits is not None:
            self.set_fits(path_to_fits)
        if path_to_samples is not None:
            self.set_samples(path_to_samples)

    def set_fits(self, file: str):
        """Set GWFits object."""
        self.fits = GWFits(file)
        logging.getLogger(self.logger).info(
            "[GW] Fits is loaded from the file %s", os.path.basename(file)
        )
        self.utc = self.fits.utc
        self.jd = self.fits.jd

    def set_samples(self, file: str):
        """Set GWSamples object."""
        self.samples = GWSamples(file)
        logging.getLogger(self.logger).info(
            "[GW] Samples are loaded from the file %s", os.path.basename(file)
        )


class GWFits:
    def __init__(self, file):
        assert os.path.isfile(file)
        self.file = file
        header = fits.read_sky_map(self.file, nest=False)[1]
        self.utc = astropy.time.Time(header["gps_time"], format="gps").utc
        self.jd = jang.conversions.utc_to_jd(self.utc)

    def get_skymap(self, nside: int = None) -> np.ndarray:
        """Get the skymap from FITS file."""
        skymap = fits.read_sky_map(self.file, nest=False)[0]
        if nside is not None:
            skymap = hp.pixelfunc.ud_grade(skymap, nside)
            skymap *= 1 / np.sum(skymap)
        return skymap

    def get_signal_region(self, nside: int = None, contained_prob: float = 0.90) -> np.ndarray:
        """Get the region containing a given probability of the skymap, for a given resolution."""
        skymap = self.get_skymap(nside)
        npix = hp.get_map_size(skymap)
        nside = hp.npix2nside(npix)

        if contained_prob is None:
            return np.arange(npix)

        iSort = np.flipud(np.argsort(skymap))
        sortedCumulProba = np.cumsum(skymap[iSort])
        cumulProba = np.empty_like(sortedCumulProba)
        cumulProba[iSort] = sortedCumulProba

        pixReg = np.arange(npix)
        pixReg = pixReg[cumulProba <= contained_prob]
        return pixReg


class GWSamples:
    def __init__(self, file):
        assert os.path.isfile(file)
        self.file = file
        self.sample_name = None
        self.mass1 = None
        self.mass2 = None

    def find_correct_sample(self) -> str:
        """Find the correct posterior samples, first available one
        from the list in the global variable 'priorities' in gw.py.
        """
        f = h5py.File(self.file, "r")
        keys = list(f.keys())
        f.close()
        for sample in priorities:
            if sample in keys:
                self.sample_name = sample
                break
        if self.sample_name is None:
            raise RuntimeError(f"Did not find a correct sample in {self.file}")
        return self.sample_name

    def get_variables(self, *variables) -> dict:
        """Get posterior samples from the GW data release."""

        if self.sample_name is None:
            self.find_correct_sample()
        f = h5py.File(self.file, "r")

        variables_h5 = f[self.sample_name]["posterior_samples"][:].dtype.names
        variables_corrected = []
        for var in variables:
            if var not in variables_h5:
                if (var + "_non_evolved") in variables_h5:
                    variables_corrected.append(var + "_non_evolved")
                else:
                    raise RuntimeError("Missing variable %s in h5 file." % var)
            else:
                variables_corrected.append(var)
        data = {}
        for var, varC in zip(variables, variables_corrected):
            data[var] = f[self.sample_name]["posterior_samples"][varC]
        f.close()

        return data

    @cached_property
    def masses(self) -> Tuple[float, float]:
        """Tuple of the two source masses."""
        masses = self.get_variables("mass_1_source", "mass_2_source")
        self.mass1 = np.mean(masses["mass_1_source"])
        self.mass2 = np.mean(masses["mass_2_source"])
        return self.mass1, self.mass2

    @cached_property
    def distance_mean(self) -> float:
        """Mean luminosity distance based on posterior samples."""
        dist = self.get_variables("luminosity_distance")
        return np.mean(dist["luminosity_distance"])

    @cached_property
    def distance_error(self) -> float:
        """Mean luminosity distance based on posterior samples."""
        dist = self.get_variables("luminosity_distance")
        return np.std(dist["luminosity_distance"])

    @cached_property
    def type(self) -> str:
        """Most probable type of GW event"""
        mass1, mass2 = self.masses
        if self.mass1 > 3:
            if self.mass2 > 3:
                return "BBH"
            return "NSBH"
        return "BNS"

    def prepare_toys(
        self, *variables, nside: int, region_restriction: Optional[np.ndarray] = None
    ) -> dict:
        """Prepare GW toys with an eventual restriction to the considered sky region."""
        toys = self.get_variables(*variables)
        toys["ipix"] = hp.ang2pix(nside, np.pi / 2 - toys["dec"], toys["ra"])

        if region_restriction is not None:
            to_keep = [
                i for i, pix in enumerate(toys["ipix"]) if pix in region_restriction
            ]
            for k in toys.keys():
                toys[k] = toys[k][to_keep]
        return toys


class Database:
    """Database containing all GW events."""

    def __init__(
        self, filepath: Optional[str] = None, db: Optional[pd.DataFrame] = None, name: Optional[str] = None
    ):
        self._filepath = filepath
        self.db = db
        self.name = name
        if filepath is not None and db is not None:
            logging.getLogger("jang").warning(
                "Both a file path and a DataFrame have been passed to Database, will use the database and ignore the file."
            )
        elif filepath is not None:
            if not os.path.isfile(filepath):
                logging.getLogger("jang").warning("Input files does not exist, starting from empty database.")
            self.db = pd.read_csv(filepath, index_col=0)
            if self.name is None:
                self.name = os.path.splitext(os.path.basename(filepath))[0]

    def add_entry(self, name: str, h5path: str, fitspath: str):
        entry = {"h5_filepath": h5path, "fits_filepath": fitspath}
        newline = pd.DataFrame([entry], index=[name])
        if self.db is None:
            self.db = newline
        else:
            self.db = self.db.append(newline)

    def find_gw(self, name: str):
        if name not in self.db.index:
            raise RuntimeError("[gw.Database] Missing index %s in the database." % name)
        gw = GW(
            name,
            path_to_samples=self.db.loc[name]["h5_filepath"],
            path_to_fits=self.db.loc[name]["fits_filepath"],
        )
        if self.name is not None:
            gw.catalog = self.name
        return gw

    def list_all(self):
        return list(self.db.index)

    def list(
        self,
        gwtype: Optional[str] = None,
        mindist: Optional[float] = None,
        maxdist: Optional[float] = None,
    ):
        selected_gw = []
        for idx, ev in self.db.iterrows():
            gw = GW(
                idx,
                path_to_samples=ev["h5_filepath"],
                path_to_fits=ev["fits_filepath"],
                logger="",
            )
            selected = True
            if gwtype is not None and gw.samples.type != gwtype:
                selected = False
            if mindist is not None and gw.samples.distance_mean < mindist:
                selected = False
            if maxdist is not None and gw.samples.distance_mean > maxdist:
                selected = False
            if selected:
                selected_gw.append(idx)
        return selected_gw

    def save(self, filepath: Optional[str] = None):
        """Save the database to specified CSV or, by default, to the one defined when initialising the Database."""
        outfile = None
        if filepath is not None:
            outfile = filepath
        elif self._filepath is not None:
            outfile = self._filepath
        else:
            raise RuntimeError("[gw.Database] No output file was provided.")
        self.db.sort_index(inplace=True)
        self.db.to_csv(outfile)


def get_search_region(detector: jang.neutrinos.Detector, gw: GW, parameters: jang.parameters.Parameters):

    region = gw.fits.get_signal_region(
        parameters.nside, parameters.get_searchregion_gwfraction()
    )
    if not parameters.get_searchregion_iszeroincluded():
        region_nonzero = detector.get_nonempty_acceptance_pixels(
            parameters.spectrum, parameters.nside
        )
        region = np.intersect1d(region, region_nonzero)

    return region
