import numpy as np
from typing import Union, Optional


def infer_uncertainties(input_array: Union[float, np.ndarray], nsamples: int, correlation: Optional[float] = None) -> np.ndarray:
    """Infer uncertainties based on an input array that could be:
        - 0-D (same error for each sample)
        - 1-D (one error per sample)
        - 2-D (correlation matrix)
    """
    if input_array is None:
        return None
    input_array = np.array(input_array)
    correlation_matrix = (correlation if correlation is not None else 0) * np.ones((nsamples, nsamples))
    np.fill_diagonal(correlation_matrix, 1)
    # if uncertainty is a scalar (error for all samples)
    if input_array.ndim == 0:
        return input_array * correlation_matrix * input_array
    # if uncertainty is a vector (error for each sample)
    if input_array.shape == (nsamples,):
        return np.array([[input_array[i] * correlation_matrix[i, j] * input_array[j] for i in range(nsamples)] for j in range(nsamples)])
    # if uncertainty is a covariance matrix
    if input_array.shape == (nsamples, nsamples):
        return input_array
    raise RuntimeError("The size of uncertainty_acceptance does not match with the number of samples")


def get_search_region(detector: 'jang.io.NuDetector', gw: 'jang.io.GW', pars: 'jang.io.Parameters'):
    """Find the search region for the joint GW+nu signal based on a given GW event, list of neutrino samples and configuration>
    In the configuration, the entry `search_region` is used to define the search region with respect """

    region = gw.fits.get_signal_region(pars.nside, pars.get_searchregion_gwfraction())
    if not pars.get_searchregion_iszeroincluded():
        region_nonzero = detector.get_nonempty_acceptance_pixels(pars.spectrum, pars.nside)
        region = np.intersect1d(region, region_nonzero)
    if len(region) == 0:
        raise RuntimeError("The search region has been reduced to empty. Please check 'search_region' parameter.")
    return region
