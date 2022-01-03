"""Compute upper limits for any given Detector or SuperDetector and selected model, using the likelihoods."""

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Union

from jang.gw import GW
from jang.neutrinos import Detector, SuperDetector
from jang.parameters import Parameters
import jang.posteriors

matplotlib.use("Agg")


def get_limit_flux(
    detector: Union[Detector, SuperDetector],
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on flux normalization and save the related likelihood in files "saved_lkl.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_flux_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plt.close("all")
        plt.plot(x, y)
        plt.xscale("log")
        plt.xlabel(r"Flux normalisation")
        plt.ylabel("Posterior")
        plt.savefig(saved_lkl + ".png", dpi=300)
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(Flux) = %.3e",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.spectrum,
        limit,
    )
    return limit


def get_limit_etot(
    detector: Union[Detector, SuperDetector],
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on E(tot) and save the related likelihood in files "saved_lkl.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_etot_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plt.close("all")
        plt.plot(x, y)
        plt.xscale("log")
        plt.xlabel(r"$E^{tot}_{\nu}$")
        plt.ylabel("Posterior")
        plt.savefig(saved_lkl + ".png", dpi=300)
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(Etot) = %.3e",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.spectrum,
        limit,
    )
    return limit


def get_limit_fnu(
    detector: Union[Detector, SuperDetector],
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on fnu=E(tot)/E(radiated) and save the related likelihood in files "saved_lkl.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_fnu_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plt.close("all")
        plt.plot(x, y)
        plt.xscale("log")
        plt.xlabel(r"$E^{tot}_{\nu}/E_{rad}$")
        plt.ylabel("Posterior")
        plt.savefig(saved_lkl + ".png", dpi=300)
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, limit(fnu) = %.3e",
        gw.name,
        detector.name,
        parameters.spectrum,
        limit,
    )
    return limit
