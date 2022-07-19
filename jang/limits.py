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
import jang.stats
import jang.stats.posteriors

matplotlib.use("Agg")


def plot_posterior(
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
    outfile: Optional[str] = None,
) -> float:
    """Return 90% upper limit on flux normalization (dn/dE = norm*{parameters.spectrum}).
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("flux", parameters.range_flux[0:2], parameters.range_flux[2], log=True)]

    x, y = jang.stats.posteriors.compute_flux_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, "Flux normalization", outfile + ".png")
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
    outfile: Optional[str] = None,


) -> float:
    """Return 90% upper limit on the total energy emitted in neutrinos (all-flavours) E(tot) [in erg].
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("etot", parameters.range_etot[0:2], parameters.range_etot[2], log=True)]

    x, y = jang.stats.posteriors.compute_etot_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, r"$E^{tot}_{\nu}$", outfile + ".png")
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
    outfile: Optional[str] = None,
) -> float:
    """Return 90% upper limit on fnu=E(tot)/E(radiated).
    The related posterior will be saved in "{outfile}.[npy/png]" if provided."""

    variables = [jang.stats.PosteriorVariable("fnu", parameters.range_fnu[0:2], parameters.range_fnu[2], log=True)]

    x, y = jang.stats.posteriors.compute_fnu_posterior(variables, detector, gw, parameters)
    x, y = jang.stats.normalize(x[0], y)
    limit = jang.stats.compute_upperlimit_from_x_y(x, y)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.save(outfile + ".npy", [x, y])
        plot_posterior(x, y, limit, r"$E^{tot}_{\nu}/E_{rad}$", outfile + ".png")
    logging.getLogger("jang").info(
        "[Limits] %s, %s, %s, %s, limit(fnu) = %.3e",
        gw.name,
        detector.name,
        parameters.jet.__repr__(),
        parameters.spectrum,
        limit,
    )
    return limit
