"""Computation of significance."""

import logging
import numpy as np

from jang.analysis import Analysis
from jang.gw import GW, get_search_region
from jang.neutrinos import Detector
from jang.parameters import Parameters
from jang.posteriors import poisson_several_samples, prior_signal


def compute_prob_null_hypothesis(detector: Detector, gw: GW, parameters: Parameters):

    ana = Analysis(gw=gw, detector=detector, parameters=parameters)
    ana.prepare_toys()

    x_arr = np.logspace(*parameters.range_flux)
    F0, F1 = 0, 0
    gamma, delta = 0, 0

    for toy in ana.toys:
        phi_to_nsig = ana.phi_to_nsig(toy)
        all_zeros = np.zeros_like(toy[1].nbackground)
        # H1 (signal+background) // Nobs=0
        F1 += np.sum(
            poisson_several_samples(
                all_zeros, toy[1].nbackground, phi_to_nsig, x_arr[:-1],
            )
            * prior_signal(
                x_arr[:-1], toy[1].nbackground, phi_to_nsig, parameters.prior_signal
            )
            * np.diff(x_arr)
        )
        # H1 (signal+background) // Nobs=real
        delta += np.sum(
            poisson_several_samples(
                toy[1].nobserved, toy[1].nbackground, phi_to_nsig, x_arr[:-1],
            )
            * prior_signal(
                x_arr[:-1], toy[1].nbackground, phi_to_nsig, parameters.prior_signal
            )
            * np.diff(x_arr)
        )

    for toy in ana.toys_det:
        all_zeros = np.zeros_like(toy.nbackground)
        # H0 (background) // Nobs=0
        F0 += poisson_several_samples(all_zeros, toy.nbackground, all_zeros, 0.0,)
        # H0 (background) // Nobs=real
        gamma += poisson_several_samples(toy.nobserved, toy.nbackground, all_zeros, 0.0)

    P0 = gamma / (gamma + F0 / F1 * delta)

    logging.getLogger("jang").info(
        "[Significance] %s, %s, %s, P(H0 | data) = %.3e %%",
        gw.name,
        detector.name,
        parameters.spectrum,
        100 * P0,
    )
    return P0
