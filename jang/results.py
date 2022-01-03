"""Formatting the results in a Pandas database and related plotting functions."""

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Optional, Union

from jang.conversions import JetModelBase
from jang.gw import GW
from jang.neutrinos import Detector, SuperDetector
from jang.parameters import Parameters

matplotlib.use("Agg")


class Database:
    """Database containing all obtained results."""

    def __init__(
        self, filepath: Optional[str] = None, db: Optional[pd.DataFrame] = None
    ):
        self._filepath = filepath
        self.db = db
        if filepath is not None and db is not None:
            logging.getLogger("jang").warning(
                "Both a file path and a DataFrame have been passed to Database, will use the database and ignore the file."
            )
        elif filepath is not None and os.path.isfile(filepath):
            self.db = pd.read_csv(filepath, index_col=0)

    def add_entry(
        self,
        detector: Union[Detector, SuperDetector],
        gw: GW,
        parameters: Parameters,
        limit_flux: float,
        limit_etot: float,
        limit_fnu: float,
        file_lkl_flux: str,
        file_lkl_etot: str,
        file_lkl_fnu: str,
    ):
        """Adds new entry to the database."""
        sample_names = []
        if isinstance(detector, Detector):
            sample_names = [s.shortname for s in detector.samples]
        elif isinstance(detector, SuperDetector):
            sample_names = [
                "%s/%s" % (det.name, s.shortname)
                for det in detector.detectors
                for s in det.samples
            ]
        else:
            raise TypeError("Unsupported object as detector.")
        entry = {
            "Detector.name": detector.name,
            "Detector.samples": sample_names,
            "Detector.error_acceptance": list(
                np.sqrt(np.diag(detector.error_acceptance))
            )
            if parameters.apply_det_systematics
            else "",
            "Detector.results.nobserved": [s.nobserved for s in detector.samples],
            "Detector.results.nbackground": [s.background for s in detector.samples],
            "GW.name": gw.name,
            "GW.mass1": gw.samples.masses[0],
            "GW.mass2": gw.samples.masses[1],
            "GW.type": gw.samples.type,
            "GW.distance_mean": gw.samples.distance_mean,
            "GW.distance_error": gw.samples.distance_error,
            "Parameters.systematics": "on"
            if parameters.apply_det_systematics
            else "off",
            "Parameters.systematics.ntoys": parameters.ntoys_det_systematics
            if parameters.apply_det_systematics
            else "",
            "Parameters.energy_range": parameters.range_energy_integration,
            "Parameters.neutrino_spectrum": parameters.spectrum,
            "Parameters.jet_model": parameters.jet.__repr__(),
            "Results.limit_flux": limit_flux,
            "Results.limit_etot": limit_etot,
            "Results.limit_fnu": limit_fnu,
            "Results.likelihoods_flux": file_lkl_flux,
            "Results.likelihoods_etot": file_lkl_etot,
            "Results.likelihoods_fnu": file_lkl_fnu,
        }
        newline = pd.DataFrame([entry])
        if self.db is None:
            self.db = newline
        else:
            self.db = self.db.append(newline, ignore_index=True)

    def save(self, filepath: Optional[str] = None):
        """Save the datavase to specified CSV or, by default, to the one defined when initialising the Database."""
        outfile = None
        if filepath is not None:
            outfile = filepath
        elif self._filepath is not None:
            outfile = self._filepath
        else:
            raise RuntimeError("No output file was provided.")
        self.db = self.db[sorted(self.db.columns)]
        self.db.to_csv(outfile)

    def select_detector(self, det: Union[Detector, SuperDetector, str]):
        """Select a given detector and return a reduced database."""
        if isinstance(det, str):
            return Database(db=self.db[self.db["Detector.name"] == det])
        else:
            return Database(db=self.db[self.db["Detector.name"] == det.name])

    def select_spectrum(self, spectrum: str):
        """Select a given neutrino spectrum."""
        return Database(db=self.db[self.db["Parameters.neutrino_spectrum"] == spectrum])

    def select_jetmodel(self, jetmodel: Union[JetModelBase, str]):
        """Select a given jet model."""
        if isinstance(jetmodel, str):
            return Database(db=self.db[self.db["Parameters.jet_model"] == jetmodel])
        else:
            return Database(
                db=self.db[self.db["Parameters.jet_model"] == jetmodel.__repr__()]
            )

    def select(
        self,
        det: Optional[Union[Detector, SuperDetector, str]] = None,
        spectrum: Optional[str] = None,
        jetmodel: Optional[Union[JetModelBase, str]] = None,
    ):
        if det is None and spectrum is None and jetmodel is None:
            logging.getLogger("jang").info(
                "No selection is applied, return the infiltered database."
            )
        db = self
        if det is not None:
            db = db.select_detector(det)
        if spectrum is not None:
            db = db.select_spectrum(spectrum)
        if jetmodel is not None:
            db = db.select_jetmodel(jetmodel)
        return db

    def plot_energy_vs_distance(self, outfile: str, cat: Optional[dict] = None):
        """Draw the distribution of the obtained limits, for a given selection of entries, as a function of luminosity distance."""

        plt.close("all")
        if cat is None:
            plt.errorbar(
                x=self.db["GW.distance_mean"],
                y=self.db["Results.limit_etot"],
                xerr=self.db["GW.distance_error"],
                linewidth=0,
                elinewidth=1,
                marker="x",
            )
        else:
            for c, col, mar, lab in zip(
                cat["categories"], cat["colors"], cat["markers"], cat["labels"]
            ):
                db_cat = self.db[self.db[cat["column"]] == c]
                plt.errorbar(
                    x=db_cat["GW.distance_mean"],
                    y=db_cat["Results.limit_etot"],
                    xerr=db_cat["GW.distance_error"],
                    linewidth=0,
                    elinewidth=1,
                    color=col,
                    marker=mar,
                    label=lab,
                )
            plt.legend(loc="upper left")
        plt.xlabel("distance [Mpc]")
        plt.ylabel(r"$E^{90\%}_{tot,\nu}$ [erg]")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
