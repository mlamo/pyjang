"""Running an example analysis to do some tests."""

import tempfile
import unittest
import healpy as hp
import numpy as np

import jang.conversions
import jang.gw
import jang.limits
import jang.results
from jang.neutrinos import BackgroundFixed, Detector
from jang.parameters import Parameters


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            analysis:
              nside: 8
              apply_det_systematics: 0
              ntoys_det_systematics: 0
              fraction_of_gwregion: 0.9
              prior_signal: flat

            range:
              log10_flux: [-5, 5, 1000]
              log10_etot: [48, 62, 1400]
              log10_fnu: [-5, 10, 1500]
              neutrino_energy_GeV: [0.1, 1e8]
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        #
        detector_str = """
            name: TestDet

            nsamples: 2
            samples:
              names: ["sampleA", "sampleB"]
              shortnames: ["A", "B"]
              energyrange: [0, 100]

            earth_location:
              latitude: 10.0
              longitude: 50.0
              units: deg

            errors:
              acceptance: 0.00
              acceptance_corr: 1
              background: 0.00
        """
        self.det_file = f"{self.tmpdir}/detector.yaml"
        with open(self.det_file, "w") as f:
            f.write(detector_str)
        self.accs = [np.ones(hp.nside2npix(8)), np.ones(hp.nside2npix(8))]
        #
        self.gwdb_file = "examples/input_files/gw_catalogs/database_example.csv"
        self.db_file = f"{self.tmpdir}/db.csv"

    def test_example(self):
        # configuration
        pars = Parameters(self.config_file)
        pars.set_models("x**-2", jang.conversions.JetIsotropic())
        # GW database
        database_gw = jang.gw.Database(self.gwdb_file)
        gw = database_gw.find_gw("GW190412")
        # detector
        det = Detector(self.det_file)
        det.set_acceptances(self.accs, pars.spectrum, pars.nside)
        bkg = [BackgroundFixed(b) for b in [0.1, 0.3]]
        det.set_observations([0, 0], bkg)
        # compute limits
        limit_flux = jang.limits.get_limit_flux(det, gw, pars, f"{self.tmpdir}/flux")
        limit_etot = jang.limits.get_limit_etot(det, gw, pars, f"{self.tmpdir}/etot")
        limit_fnu = jang.limits.get_limit_fnu(det, gw, pars, f"{self.tmpdir}/fnu")
        # save in database
        database_res = jang.results.Database(self.db_file)
        database_res.add_entry(
            det, gw, pars, limit_flux, limit_etot, limit_fnu, None, None, None
        )
        database_res.save()
        # open database and make plot
        database_res = jang.results.Database(self.db_file)
        with self.assertLogs(level="INFO"):
            database_res = database_res.select()
        database_res = database_res.select(det, pars.spectrum, pars.jet)
        database_res.plot_energy_vs_distance(f"{self.tmpdir}/eiso.png")
        cat = {
            "column": "GW.type",
            "categories": ["BBH", "BNS", "NSBH"],
            "labels": ["BBH", "BNS", "NSBH"],
            "colors": ["black", "blue", "orange"],
            "markers": ["+", "x", "^"],
        }
        database_res.plot_energy_vs_distance(f"{self.tmpdir}/eiso.png", cat=cat)
