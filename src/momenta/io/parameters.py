"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import yaml
from typing import Optional

from momenta.io.gw import default_samples_priorities
from momenta.utils.flux import FluxBase
from momenta.utils.conversions import JetModelBase


class Parameters:
    def __init__(self, file: Optional[str] = None):
        self.file = None
        self.flux = None
        self.jet = None
        self.gw_posteriorsamples_priorities = default_samples_priorities
        if file is not None:
            assert os.path.isfile(file)
            self.file = file
            with open(self.file, "r") as f:
                params = yaml.safe_load(f)
            # analysis parameters
            self.nside = params.get("skymap_resolution")
            self.apply_det_systematics = bool(params["detector_systematics"])
            self.likelihood_method = params["analysis"]["likelihood"]
            # signal priors
            self.prior_normalisation = params["analysis"]["prior_normalisation"]["type"]
            self.prior_normalisation_range = params["analysis"]["prior_normalisation"]["range"]
            # GW parameters
            if "gw" in params and "sample_priorities" in params["gw"]:
                self.gw_posteriorsamples_priorities = params["gw"]["sample_priorities"]

    def set_models(self, flux: FluxBase = None, jet: JetModelBase = None):
        """Set the neutrino flux model and jet model."""
        if flux is not None:
            self.flux = flux
        if jet is not None:
            self.jet = jet

    @property
    def str_filename(self):
        """Get the representation of the parameters in string format for suffixing filenames."""
        str_model = []
        if self.flux is not None:
            str_model.append(str(self.flux))
        if self.jet is not None:
            str_model.append(self.jet.str_filename)
        return "_".join(str_model)

    def get_searchregion_gwfraction(self) -> float:
        spl = self.search_region.split("_")
        if len(spl) >= 2 and spl[0] == "region":
            return float(spl[1]) / 100
        if len(spl) == 1 and spl[0] == "bestfit":
            return 0
        if len(spl) == 1 and spl[0] == "fullsky":
            return None
        return None

    def get_searchregion_iszeroincluded(self) -> bool:
        """Returns True if the pixels with zero acceptance should be included."""
        spl = self.search_region.split("_")
        if spl[-1] == "excludezero":
            return False
        return True
