"""Computation of posteriors."""

import numpy as np
from typing import Tuple

from jang.gw import GW
from jang.neutrinos import Detector
from jang.parameters import Parameters
from jang.analysis import Analysis
import jang.stats.likelihoods as lkl
import jang.stats.priors as prior


def compute_flux_posterior(detector: Detector, gw: GW, parameters: Parameters, other_variables: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of all-flavour neutrino flux at Earth.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable flux
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)

    x_arr = np.logspace(*parameters.range_flux)
    if other_variables is not None:
        x_arr, *y_arr = np.meshgrid(x_arr, *other_variables.values(), indexing='ij')
        ys_arr = {}
        for i, var in enumerate(other_variables.keys()):
            ys_arr[var] = y_arr[i]

    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            post_arr += lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, phi_to_nsig, x_arr, ys_arr) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, phi_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            post_arr += lkl.pointsource_several_samples(detector.samples, toy[1].nobserved, toy[1].nbackground, phi_to_nsig, toy[0].ra, toy[0].dec, x_arr, ys_arr) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, phi_to_nsig, parameters.prior_signal)

    print(x_arr.shape, post_arr.shape)
    exit()
    return x_arr, post_arr


def compute_etot_posterior(detector: Detector, gw: GW, parameters: Parameters) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of total energy.

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable Etot
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn")

    x_arr = np.logspace(*parameters.range_etot)
    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        etot_to_nsig = ana.etot_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            post_arr += lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, etot_to_nsig, x_arr) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, etot_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            post_arr += lkl.pointsource_several_samples(detector.samples, toy[1].nobserved, toy[1].nbackground, etot_to_nsig, x_arr, toy[0].ra, toy[0].dec) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, etot_to_nsig, parameters.prior_signal)
    return x_arr, post_arr


def compute_fnu_posterior(detector: Detector, gw: GW, parameters: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the posterior as a function of fnu=E(tot)/E(radiated).

    Args:
        detector (Detector): holds the nominal results
        gw (GW): holds the gravitational wave information
        parameters (Parameters): holds the needed parameters (skymap resolution to be used, neutrino spectrum and integration range...)

    Returns:
        np.ndarray: array of the variable fnu
        np.ndarray: array of computed posterior
    """

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.add_gw_variables("luminosity_distance", "theta_jn", "radiated_energy")

    x_arr = np.logspace(*parameters.range_fnu)
    post_arr = np.zeros_like(x_arr)

    for toy in ana.toys:
        fnu_to_nsig = ana.fnu_to_nsig(toy)
        if parameters.likelihood_method == "poisson":
            post_arr += lkl.poisson_several_samples(toy[1].nobserved, toy[1].nbackground, fnu_to_nsig, x_arr) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, fnu_to_nsig, parameters.prior_signal)
        elif parameters.likelihood_method == "pointsource":
            post_arr += lkl.pointsource_several_samples(detector.samples, toy[1].nobserved, toy[1].nbackground, fnu_to_nsig, x_arr, toy[0].ra, toy[0].dec) * \
                prior.signal_parameter(x_arr, toy[1].nbackground, fnu_to_nsig, parameters.prior_signal)
    return x_arr, post_arr
