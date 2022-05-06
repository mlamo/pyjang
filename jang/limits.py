"""Compute upper limits for any given Detector or SuperDetector (combination of Detector) and selected model, using the likelihoods."""

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Union

from jang.gw import GW
from jang.neutrinos import DetectorBase
from jang.parameters import Parameters
import jang.posteriors

matplotlib.use("Agg")


def plot_likelihood(
    x: np.ndarray, y: np.ndarray, limit: float, xlabel: str, outfile: str
):

    plt.close("all")
    plt.plot(x, y)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Posterior")
    plt.axvline(limit, color="red", label=r"90% upper limit")
    plt.legend(loc="upper right")
    plt.savefig(outfile, dpi=300)


def get_limit_flux(
    detector: DetectorBase,
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on flux normalization (dn/dE = norm*{parameters.spectrum}).
    The related likelihood will be saved in "{saved_lkl}.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_flux_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plot_likelihood(x, y, limit, "Flux normalization", saved_lkl + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, limit(Flux) = %.3e",
        gw.name,
        detector.name,
        parameters.spectrum,
        limit,
    )
    return limit


def get_limit_etot(
    detector: DetectorBase,
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on the total energy emitted in neutrinos (all-flavours) E(tot) [in erg].
    The related likelihood will be saved in "{saved_lkl}.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_etot_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plot_likelihood(x, y, limit, r"$E^{tot}_{\nu}$", saved_lkl + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(Etot) = %.3e erg",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.spectrum,
        limit,
    )
    return limit


def get_limit_fnu(
    detector: DetectorBase,
    gw: GW,
    parameters: Parameters,
    saved_lkl: Optional[str] = None,
) -> float:
    """Return 90% upper limit on fnu=E(tot)/E(radiated).
    The related likelihood will be saved in "{saved_lkl}.[npy/png]" if provided."""
    x, y = jang.posteriors.compute_fnu_posterior(detector, gw, parameters)
    x, y = jang.posteriors.normalize(x, y)
    limit = jang.posteriors.compute_upperlimit_from_x_y(x, y)
    if saved_lkl is not None:
        os.makedirs(os.path.dirname(saved_lkl), exist_ok=True)
        np.save(saved_lkl + ".npy", [x, y])
        plot_likelihood(x, y, limit, r"$E^{tot}_{\nu}/E_{rad}$", saved_lkl + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(fnu) = %.3e",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.spectrum,
        limit,
    )
    return limit
