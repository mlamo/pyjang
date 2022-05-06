"""Computation of posteriors."""

import numpy as np
from scipy.special import factorial
from scipy.interpolate import interp1d
from typing import Tuple

from jang.gw import GW, get_search_region
from jang.neutrinos import Detector
from jang.parameters import Parameters
import jang.conversions


def compute_upperlimit_from_x_y(
    x_arr: np.ndarray, y_arr: np.ndarray, CL: float = 0.9
) -> float:
    """Compute the upper limit at a confidence level CL for a given posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return np.inf
    int_arr = y_arr[:-1] * (x_arr[1:] - x_arr[:-1])
    cum_arr = np.cumsum(int_arr)
    cum_arr = 1 / cum_arr[-1] * cum_arr
    f = interp1d(cum_arr, x_arr[:-1])
    limit = float(f(CL))
    # limit dangerously closed to lkl upper bound
    if limit > 0.9 * CL * x_arr[-1]:
        return np.inf
    return limit


def normalize(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return x_arr, y_arr
    integral = np.sum(y_arr[:-1] * (x_arr[1:] - x_arr[:-1]))
    return x_arr, 1 / integral * y_arr


def poisson_one_sample(
    nobserved: int, nbackground: float, conv: float, var: np.ndarray
) -> np.ndarray:
    """Compute the likelihood Poisson(n_observed, n_background + conv * var) as a function of var."""
    nexpected = conv * var + nbackground
    return np.power(nexpected, nobserved) / factorial(nobserved) * np.exp(-nexpected)


def poisson_several_samples(
    nobserved: np.ndarray,
    nbackground: np.ndarray,
    conv: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Compute the multi-sample Poisson lkl. Each argument are arrays with one entry per sample."""
    lkl = np.ones_like(var)
    for n_obs, n_bkg, cv in zip(nobserved, nbackground, conv):
        lkl *= poisson_one_sample(n_obs, n_bkg, cv, var)
    return lkl


def prior_signal(var: np.ndarray, bkg: np.ndarray, conv: np.ndarray, prior_type: str):

    if prior_type == "flat":
        return np.ones_like(var)
    elif prior_type == "jeffrey":
        nsamples = len(bkg)
        tmp = [
            conv[i] ** 2 / (conv[i] * var + bkg[i]) if conv[i] > 0 else 0
            for i in range(nsamples)
        ]
        return np.sqrt(np.sum(tmp, axis=0))
    else:
        raise RuntimeError(f"Unknown prior type {prior_type}")


def compute_flux_posterior(
    detector: Detector, gw: GW, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of all-flavour neutrino flux at Earth.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable flux
        np.ndarray: array of computed posterior
    """

    acceptances, nside = detector.get_acceptances(parameters.spectrum)
    if parameters.nside is None:
        parameters.nside = nside
    elif parameters.nside != nside:
        raise RuntimeError("Something went wrong with map resolutions!")

    region_restricted = get_search_region(detector, gw, parameters)
    toys_gw = gw.samples.prepare_toys(
        "ra",
        "dec",
        nside=parameters.nside,
        region_restriction=region_restricted,
    )
    ntoys_gw = len(toys_gw["ipix"])

    if parameters.apply_det_systematics:
        ntoys_det = parameters.ntoys_det_systematics
        toys_det = detector.prepare_toys(ntoys_det)
    else:
        ntoys_det = 1
        toys_det = detector.prepare_toys(0)
    x_arr = np.logspace(*parameters.range_flux)
    post_arr = np.zeros_like(x_arr)

    for idet in range(ntoys_det):
        for igw in range(ntoys_gw):
            ipix = toys_gw["ipix"][igw]
            phi_to_nsig = np.array(
                [
                    acc.evaluate(ipix, nside=parameters.nside)
                    for i, acc in enumerate(acceptances)
                ]
            )
            phi_to_nsig *= toys_det[idet].var_acceptance
            phi_to_nsig /= 6
            post_arr += poisson_several_samples(
                toys_det[idet].nobserved,
                toys_det[idet].nbackground,
                phi_to_nsig,
                x_arr,
            ) * prior_signal(
                x_arr,
                toys_det[idet].nbackground,
                phi_to_nsig,
                parameters.prior_signal,
            )
    return x_arr, post_arr


def compute_etot_posterior(
    detector: Detector, gw: GW, parameters: Parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of total energy.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable Etot
        np.ndarray: array of computed posterior
    """

    acceptances, nside = detector.get_acceptances(parameters.spectrum)
    if parameters.nside is None:
        parameters.nside = nside
    elif parameters.nside != nside:
        raise RuntimeError("Something went wrong with map resolutions!")

    region_restricted = get_search_region(detector, gw, parameters)
    toys_gw = gw.samples.prepare_toys(
        "ra",
        "dec",
        "luminosity_distance",
        "theta_jn",
        nside=parameters.nside,
        region_restriction=region_restricted,
    )
    ntoys_gw = len(toys_gw["ipix"])

    if parameters.apply_det_systematics:
        ntoys_det = parameters.ntoys_det_systematics
        toys_det = detector.prepare_toys(ntoys_det)
    else:
        ntoys_det = 1
        toys_det = detector.prepare_toys(0)
    x_arr = np.logspace(*parameters.range_etot)
    post_arr = np.zeros_like(x_arr)

    for idet in range(ntoys_det):
        for igw in range(ntoys_gw):
            ipix = toys_gw["ipix"][igw]
            phi_to_nsig = np.array(
                [
                    acc.evaluate(ipix, nside=parameters.nside)
                    for i, acc in enumerate(acceptances)
                ]
            )
            phi_to_nsig *= toys_det[idet].var_acceptance
            phi_to_nsig /= 6
            eiso_to_phi = jang.conversions.eiso_to_phi(
                parameters.range_energy_integration,
                parameters.spectrum,
                toys_gw["luminosity_distance"][igw],
            )
            etot_to_eiso = jang.conversions.etot_to_eiso(
                toys_gw["theta_jn"][igw], parameters.jet
            )
            post_arr += poisson_several_samples(
                toys_det[idet].nobserved,
                toys_det[idet].nbackground,
                etot_to_eiso * eiso_to_phi * phi_to_nsig,
                x_arr,
            ) * prior_signal(
                x_arr,
                toys_det[idet].nbackground,
                etot_to_eiso * eiso_to_phi * phi_to_nsig,
                parameters.prior_signal,
            )
    return x_arr, post_arr


def compute_fnu_posterior(
    detector: Detector, gw: GW, parameters: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of fnu=E(tot)/E(radiated).

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable fnu
        np.ndarray: array of computed posterior
    """

    acceptances, nside = detector.get_acceptances(parameters.spectrum)
    if parameters.nside is None:
        parameters.nside = nside
    elif parameters.nside != nside:
        raise RuntimeError("Something went wrong with map resolutions!")

    region_restricted = get_search_region(detector, gw, parameters)
    toys_gw = gw.samples.prepare_toys(
        "ra",
        "dec",
        "luminosity_distance",
        "theta_jn",
        "radiated_energy",
        nside=parameters.nside,
        region_restriction=region_restricted,
    )
    ntoys_gw = len(toys_gw["ipix"])

    if parameters.apply_det_systematics:
        ntoys_det = parameters.ntoys_det_systematics
        toys_det = detector.prepare_toys(ntoys_det)
    else:
        ntoys_det = 1
        toys_det = detector.prepare_toys(0)
    x_arr = np.logspace(*parameters.range_fnu)
    post_arr = np.zeros_like(x_arr)

    for idet in range(ntoys_det):
        for igw in range(ntoys_gw):
            ipix = toys_gw["ipix"][igw]
            phi_to_nsig = np.array(
                [
                    acc.evaluate(ipix, nside=parameters.nside)
                    for i, acc in enumerate(acceptances)
                ]
            )
            phi_to_nsig *= toys_det[idet].var_acceptance
            phi_to_nsig /= 6
            eiso_to_phi = jang.conversions.eiso_to_phi(
                parameters.range_energy_integration,
                parameters.spectrum,
                toys_gw["luminosity_distance"][igw],
            )
            etot_to_eiso = jang.conversions.etot_to_eiso(
                toys_gw["theta_jn"][igw], parameters.jet
            )
            fnu_to_etot = jang.conversions.fnu_to_etot(toys_gw["radiated_energy"][igw])
            post_arr += poisson_several_samples(
                toys_det[idet].nobserved,
                toys_det[idet].nbackground,
                fnu_to_etot * etot_to_eiso * eiso_to_phi * phi_to_nsig,
                x_arr,
            ) * prior_signal(
                x_arr,
                toys_det[idet].nbackground,
                fnu_to_etot * etot_to_eiso * eiso_to_phi * phi_to_nsig,
                parameters.prior_signal,
            )
    return x_arr, post_arr
