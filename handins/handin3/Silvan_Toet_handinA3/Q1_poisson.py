import numpy as np


def negative_poisson_ln_likelihood(model: callable, data: np.ndarray, params: tuple) -> float:
    """
    Calculate the Poisson negative log-likelihood for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : tuple
        The parameters to evaluate the model at.

    Returns
    -------
    float
        The Poisson negative log-likelihood value for the given parameters and data.
    """
    # TODO: implement calculation of the Poisson negative log-likelihood (or equivalent) to be minimized

    return 0.0  # replace by the correct value
