
"""Validation metrics.
Tests for homogeneity of the samples in the moving window.
"""

import numpy as np
import scipy
from dataclasses import dataclass
import datetime as dt
# from sklearn import metrics  # https://scikit-learn.org/stable/api/sklearn.metrics.html
import sklearn
import xarray as xr
import pandas as pd
import HydroErr  # https://hydroerr.readthedocs.io/en/latest/list_of_metrics.html


# Error metrics ------------------------------------------------------

def rMSE(x, y) -> float:
    """ratio mean square error

    Args: 
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series 
        y (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series to compare to x

    Returns:
        float: 
            rMSE
    """
    return np.sum((x/y - 1)**2)/len(y)


def MSE(x, y) -> float:
    """Mean square error

    Args: 
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series 
        y (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series to compare to x

    Returns:
        float: MSE
    """
    return np.sum((x - y)**2)/len(y)


def MAE(x, y) -> float:
    """Mean absolute error

    Args: 
        x (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series 
        y (array_like: Can be 1-dimensional numpy array or pandas Series with float dtypes): 
            data series to compare to x

    Returns:
        float: MAE
    """
    return np.sum(abs(x - y))/len(y)

# Compare fit and observed values ------------------------------------


def fit_vs_obs_metrics(yfit, yobs) -> dict[str, float]:
    """Returns several metrics to evaluate error between 1D
    fit and observed data.
    See https://hydroerr.readthedocs.io/en/latest/list_of_metrics.html

    Args: 
        yfit (array-like): fit/simulated data
        yobs (array-like): observed data

    Returns:
        dict[str, float]:
            Dictionary of error metrics. 
    """
    return {
        'rmse': HydroErr.rmse(yfit, yobs),  # root mean squared error
        'mse':  HydroErr.mse(yfit, yobs),  # mean sqaured error
        'mae':  HydroErr.mae(yfit, yobs),  # mean absolute error
        'ed': HydroErr.ed(yfit, yobs),  # euclidean distance
        'ned': HydroErr.ned(yfit, yobs),  # normalized euclidean distance
        'rsq': HydroErr.r_squared(yfit, yobs),
        'pearson_r': HydroErr.pearson_r(yfit, yobs),
        'spearman_r': HydroErr.spearman_r(yfit, yobs),
        'nse': HydroErr.nse(yfit, yobs),  # Nash-sutcliffe
        'nse_mod': HydroErr.nse_mod(yfit, yobs),
        'nse_rel': HydroErr.nse_rel(yfit, yobs),
        'kge_2009': HydroErr.kge_2009(yfit, yobs),  # Kling-Gupta
        'kge_2012': HydroErr.kge_2012(yfit, yobs),
        'record_length': len(yobs)  # number of values used in fit
    }


def regression_metrics(X: np.ndarray, yobs, yfit) -> dict[str, float]:
    """F-statistic and p-value for multiple linear regression fit,
    plus all fit_vs_obs_metrics.

    Args:
        X (np.ndarray): Regression matrix
        yobs (array-like): observed values
        yfit (array-like): comparabale predicted values

    Returns:
        dict[str, float]
    """
    metric_values = fit_vs_obs_metrics(yfit.squeeze(), yobs.squeeze())
    f_vals, p_vals = sklearn.feature_selection.f_regression(X, yobs.ravel())
    ind = 1
    for f, p in zip(f_vals, p_vals):
        metric_values[f'f_stat_{ind}'] = f
        metric_values[f'f_stat_p_val_{ind}'] = p
        ind += 1
    return metric_values


# Assess homogeneity of the hourly NTR samples in the moving window. --------


def homogeneity(x_sites_):
    """Applies the Hosking and Wallis (1993) test for homogeneity for a window of x_sites.

    Args:
        x_sites (list: pd.Series):
            List of times series at each site/grid cell

    Returns:

    """
    # if len(x_sites) < 2:
    if len(x_sites_) < 2:
        return None
    else:
        # Functions
        N = len(x_sites_[0])
        split = int(np.floor(N/2))
        x_sites = [x_sites_[0][:split], x_sites_[0][split:]]

        def V_site(x):
            # Get sample L-moment ratios
            L1, L2, L3, L4 = scipy.stats.lmoment(x)
            t1 = L2 / L1
            t3 = L3 / L2
            t4 = L4 / L2
            n = len(x)
            return (L1, L2, t1, t3, t4, n)

        def group_mean_V(x_list):
            vals = pd.DataFrame([V_site(x) for x in x_list],
                                columns=["L1", "L2", "t1", "t3", "t4", "n"])
            N = vals['n'].sum()
            tR_1 = (vals['t1'] * vals['n'] / N).sum()
            tR_3 = (vals['t3'] * vals['n'] / N).sum()
            tR_4 = (vals['t4'] * vals['n'] / N).sum()
            L_1 = (vals['L1'] * vals['n'] / N).sum()
            L_2 = (vals['L2'] * vals['n'] / N).sum()
            V = ((vals['t1'] - tR_1)**2 * vals['n'] / N).sum()
            return L_1, L_2, tR_1, tR_3, tR_4, N, V

        def G(z): return scipy.special.gamma(z)

        def gr(r, k, h):
            if h > 0:
                res = (r*G(1+k)*G(r/h))/(h**(1+k)*G(1+k+r/h))
            elif h < 0:
                res = (r*G(1+k)*G(-k-r/h))/((-h)**(1+k)*G(1-r/h))
            else:
                res = r**(-k) * G(1 + k)
            return res

        def func(x):
            eqns = [
                (-gr(1, x[0], x[1]) + 3*gr(2, x[0], x[1]) - 2*gr(3, x[0],
                 x[1]))/(gr(1, x[0], x[1]) - gr(2, x[0], x[1])) - tR_3,
                (-gr(1, x[0], x[1]) + 6*gr(2, x[0], x[1]) - 10*gr(3, x[0], x[1]) +
                 5*gr(4, x[0], x[1]))/(gr(1, x[0], x[1]) - gr(2, x[0], x[1])) - tR_4
            ]
            # Turn it into a single-valued function for optimization
            return np.abs(eqns[0]) + np.abs(eqns[1])

        # Computation
        # Fit Kappa distribution to the GROUP AVERAGE L-moment ratios
        L_1, L_2, tR_1, tR_3, tR_4, N, Vobs = group_mean_V(x_sites)
        x = pd.concat(x_sites)
        data_range = x.max() - x.min()
        bounds = {
            'loc': (x.min(), x.max()),
            'scale': (0, data_range),
            'k': (-1, 3),
            'h': (-1, 3),
        }
        params_mle = scipy.stats.fit(
            scipy.stats.kappa4, x, bounds, method="mle")
        # Get L-moment estimate with Newton-Raphson
        params = scipy.optimize.fmin_tnc(func, [params_mle.params.k, params_mle.params.h],
                                         approx_grad=True, bounds=[(-1, 3), (-1, 3)])
        k = params[0][0]
        h = params[0][1]
        # Estimate xi and alpha
        alpha = k*L_2/(gr(1, k, h) - gr(2, k, h))
        xi = L_1 - alpha*(1-gr(1, k, h))/k
        # Check: **k > −1; if h < 0 then hk > −1; h > −1; k + 0.725h > −1 **
        if ((h < 0) & (h*k <= -1)) | ((h > -1) & (k <= -1 - 0.725*h)):
            print("Warning: k and h estimates aren't feasible parameters")

        # Simulate the kappa world
        kdist = scipy.stats.kappa4(k, h, loc=xi, scale=alpha)
        # Stochastic sampling from kdist
        Nsim = 1000
        samps_cdf = [[np.random.random(len(site))
                      for site in x_sites] for _ in range(Nsim)]
        Xsim = [[kdist.ppf(s) for s in samp] for samp in samps_cdf]
        Vsim = [group_mean_V(samp)[-1] for samp in Xsim]
        mu_v = np.mean(Vsim)
        sd_v = np.std(Vsim)
        H = (Vobs - mu_v)/sd_v
        print(H)
        breakpoint()
        return H


def pooling(lat: float, lon: float, ts: dt.datetime, gridname: str,
            dirs: dict[str, str], ntr_window: pd.DataFrame):
    """Assessment of data pooling in the moving window:
    sample size and homogeneity.
    NOTE: Adapting the Kolomogorov-Smirnov test to assess homogeneity
    produced a test that was way too strict. 

    Args:
        ntr (pd.DataFrame):
            Hourly NTR dataset. Must have columns: x, y, time, ntr
    """
    fname = dirs.param_output_file(gridname, "pooling", lat, lon, ts)
    sample_size = ntr_window.shape[0]
    n_stations = ntr_window.loc[:, ["x", "y"]].drop_duplicates().shape[0]
    # sample_size_per_station = ntr_window.groupby(["x", "y"]).apply(lambda df: df.shape[0]).reset_index()
    xy = ntr_window[['x', 'y']].drop_duplicates()
    data_by_site = [ntr_window.loc[(ntr_window.x == x) & (ntr_window.y == y), 'ntr']
                    for x, y in zip(xy.x, xy.y)]
    h_metric = homogeneity(data_by_site)
    df = pd.DataFrame({
        'params': ['sample_size', 'n_sites', 'H'],
        'values': [sample_size, n_stations, h_metric],
    })
    df.to_csv(fname)
