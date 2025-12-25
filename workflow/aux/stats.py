
"""Distribution functions and general statistics."""

import numpy as np
import pandas as pd
import scipy
import warnings

# Distribution functions ---------------------------------------------


def distributions(dist):
    """See https://docs.scipy.org/doc/scipy/reference/scipy.stats.html"""
    return getattr(scipy.stats, dist)


def qdist(dist: str, x, **params):
    """Returns quantiles of CDF values x for distribution dist with params,
        equivalent to the scipy.stats.{dist}.ppf functions

    Args:
        dist (str):
            distribution
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            CDF values at which to compute quantiles

    Returns:
        array_like: (type matches type(x)):
            quantiles of x
    """
    return distributions(dist).ppf(x, **params)


def pdist(dist: str, x, **params):
    """Returns CDF values of x for distribution dist with params,
        equivalent to the scipy.stats.{dist}.cdf functions

    Args:
        dist (str):
            distribution
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            x values at which to compute CDF

    Returns:
        array_like: (type matches type(x)):
            CDF values at each x
    """
    return distributions(dist).cdf(x, **params)


def ddist(dist: str, x, **params):
    """Returns PDF values of x for distribution dist with params,
        equivalent to the scipy.stats.{dist}.pdf type functions

    Args:
        dist (str):
            distribution name
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            x values at which to compute PDF

    Returns:
        array_like: (type matches type(x)):
            PDF values at each x
    """
    return distributions(dist).pdf(x, **params)


def ecdf(x):
    """empirical CDF of data series x

    Args:
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes):
            data series

    Returns:
        array_like (type matches type(x)):
            CDF values evaluated at each x
    """
    x_sorted = np.sort(x)
    x_ranks = scipy.stats.rankdata(x_sorted, method='min')/(len(x)+1)
    cdf_table = pd.DataFrame({'p': x_ranks, 'value': x_sorted})
    return cdf_table


# Distribution fitting -----------------------------------------------


def distribution_shape_names(dist: str):
    distobj = distributions(dist)
    shape_names = []
    if distobj.shapes:
        shape_names = [s.strip() for s in distobj.shapes.split(',')]
    return shape_names


def distribution_param_names(dist: str):
    shape_names = distribution_shape_names(dist)
    return shape_names + ['loc', 'scale']


def distribution_fit2paramdict(dist: str, params: tuple[float]) -> dict[str, float]:
    """Names parameters based on the number of shape parameters
    for distribution dist.

    Args:
        dist (str): distribution name (see function "distributions")
        params (tuple[float]):
            tuple of unnamed parameters output from the
            scipy.stats.<distributions>.fit function

    Returns:
        dict[str, float]:
            Dictionary of named parameters
    """
    shape_params = params[:-2]
    shape_param_names = distribution_shape_names(dist)
    shape_param_dict = {
        k: v for k, v in zip(shape_param_names, shape_params)
    }
    return (
        shape_param_dict | {'loc': params[-2], 'scale': params[-1]}
    )


def fit_distribution(dist: str, x) -> dict[str, float]:
    """Fit distribution to values in x
    Args:
        dist (str): distribution name (see function "distributions")
        x (array-like): 1D vector of floats

    Returns:
        dict[str, float]:
            dictionary of parameter fits and GoF measures.
    """
    try:
        params = distributions(dist).fit(x)
    except scipy.stats._warnings_errors.FitError as exc:
        warnings.warn(
            f"Returning None, fitting distribution {dist}"
            f" failed with error {exc}"
        )
        return None
    # GoF metrics
    y_fit = pd.Series(distributions(dist).cdf(sorted(x), *params))
    y_obs = ecdf(sorted(x))['p'].loc[~y_fit.isna()]
    y_fit = y_fit[~y_fit.isna()]
    ks = scipy.stats.kstest(y_fit, y_obs)
    cvm = scipy.stats.cramervonmises(x, dist, args=params)
    return distribution_fit2paramdict(dist, params) | {
        'ks_pval': ks.pvalue,
        'cvm_pval': cvm.pvalue,
        'record_length': len(x)
    }


# ACF -----------------------------------------------


def distribution_param_names_acs(dist):
    return {
        'weibull': ['scale', 'shape'],
        'paretoII': ['scale', 'shape'],
        'burrXII': ['scale', 'shape1', 'shape2'],
        'fgn': ['H']
    }[dist]


def acf(dist: str, lag: int, **param_kwargs) -> float:
    """ACS functions to use to parametrize the autocorrelation function.

    Args:
        dist (str):
            distribution name
        lag (int):
            number of lags to compute for
        **param_kwargs:
            keyword arguments to the distribution functions

    Returns:
        float:
            autocorrelation function evaluated at lag
    """
    def weibull(scale, shape):
        return np.exp(-(lag / scale) ** shape)

    def paretoII(scale, shape):
        return (1 + (shape * lag) / scale)**((-1)/shape)

    def burrXII(scale, shape1, shape2):
        return (
            (1 + shape2 * (lag / scale) ** shape1) ** -(1 / shape1 * shape2)
        )

    def fgn(H):
        return (
            abs(-1 + lag) ** (2 * H) - 2 *
            abs(lag) ** (2 * H) + abs(1 + lag) ** (2 * H) / 2
        )

    return {
        'weibull': lambda params: weibull(**param_kwargs),
        'paretoII': lambda params: paretoII(**param_kwargs),
        'burrXII': lambda params: burrXII(**param_kwargs),
        'fgn': lambda params: fgn(**param_kwargs)
    }[dist](param_kwargs)


def distribution_param_names_acf(dist):
    lookup = {
        'weibull': ['scale', 'shape'],
        'paretoII': ['scale', 'shape'],
        'burrXII': ['scale', 'shape1', 'shape2'],
        'fgn': ['H'],
    }
    return lookup[dist]
