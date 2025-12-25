
"""Fit a smooth function to the autocorrelation structure
(up to lagmax hours)
"""

import os
import datetime as dt
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from snakemake.script import snakemake
from aux import mwm, utils, metrics


# ----------------------------------------------------------------------
# CoSMoS: fit autocorrelation function and marginal distribution
# ----------------------------------------------------------------------


def acf(dist: str, lag: int, *param_args) -> float:
    """ACS functions to use to parametrize the autocorrelation function.

    Args:
        dist (str):
            distribution name
        lag (int):
            number of lags to compute for
        *param_args:
            ordered arguments to the distribution functions (as in acf_functions)

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
        'weibull': lambda params: weibull(*param_args),
        'paretoII': lambda params: paretoII(*param_args),
        'burrXII': lambda params: burrXII(*param_args),
        'fgn': lambda params: fgn(*param_args)
    }[dist](param_args)


def fit_acs(
    dist_acf: str,
    lags: int,
    emp_acf
) -> tuple[dict[str, float], float]:
    """Estimate the parameterized ACF for CoSMoS.

    Args:
        dist_acf (str): Name of function to fit.
            Options: burrXII, weibull, paretoII
        lags (int): lags corresponding to emp_acf
        emp_acf (array-like): Empirical ACF corresponding to lags

    Returns:
        tuple[dict[str, float], float]:
            Dictionary of parameters and MAE of the fit
    """
    first_guess = {
        'burrXII': [lags, 1, 1],
        'weibull': [lags, 1],
        'paretoII': [lags, 1]
    }[dist_acf]
    bnds = {
        'burrXII': ((0.05, None), (0.05, None), (0.05, None)),
        'weibull': ((0.05, None), (0.05, None)),
        'paretoII': ((0.05, None), (0.05, None))
    }[dist_acf]

    def objective_FitACS(params):
        par_acf = np.zeros(lags)
        for lag in range(1, lags+1, 1):
            par_acf[lag - 1] = acf(dist_acf, lag, *params)
        return metrics.MAE(par_acf, emp_acf['rho'])
    result = minimize(
        objective_FitACS,
        first_guess,
        bounds=bnds,
        method='nelder-mead'
    )
    par_acs_dict = {
        'paretoII': lambda res: {'scale': res.x[0].round(2), 'shape':  res.x[1].round(2)},
        'weibull': lambda res: {'scale': res.x[0].round(2), 'shape':  res.x[1].round(2)},
        'burrXII': lambda res: {'scale': res.x[0].round(2), 'shape1': res.x[1].round(2), 'shape2': res.x[2].round(2)}
    }
    par_acs = par_acs_dict[dist_acf](result)
    gof = metrics.fit_vs_obs_metrics(
        yfit=pd.Series(
            [acf(dist_acf, lag, *result.x)
             for lag in range(1, lags+1)]
        ),
        yobs=emp_acf['rho']
    )
    return par_acs | gof


def make_lags(lag: int, dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Get pairs of data lagged by "lag" values for each dataframe in dfs,
        to be concatenated into one long dataframe of the lagged pairs.

    Args:
        lag (int):
            number of values to lag
        dfs (list[pd.DataFrame]):
            list of dataframes to lag

    Returns:
        pd.DataFrame:
            one long dataframe of the lagged pairs from each df of dfs
    """
    df_lag_1 = [df[:(df.shape[0] - lag)] for df in dfs]
    df_lag_2 = [df[(lag):] for df in dfs]
    df1 = pd.concat(df_lag_1, ignore_index=True)
    df2 = pd.concat(df_lag_2, ignore_index=True)
    df = pd.DataFrame({'left': df1['ntr'], 'right': df2['ntr']})
    return df.dropna()


def discontinuous_acs(
    df_list: list[pd.Series],
    lag_max: int
) -> np.ndarray:
    """Correctly address gaps/discontinuities in time series data
        to get lagged values.

    Args:
        df_list (list[pd.Series]):
            vector or list of vectors of continuous time series
        var (str):
            variable name with the values to autocorrelate.
        lag_max (int):
            autocorrelation lags

    Returns:
        np.ndarray:
            the empirical autocorrelation function
    """
    df_list_mod = [df.dropna()
                   for df in df_list if df.shape[0] > 0]
    df_lags = [make_lags(lag=x, dfs=df_list_mod)
               for x in range(1, lag_max+1)]
    eacs = np.array([np.corrcoef(x['left'], x['right'])[0, 1]
                     for x in df_lags])
    negative_ind = np.where(eacs < 0)[0]
    if len(negative_ind) > 0:
        eacs = eacs[:negative_ind[0]]
    return eacs


def correct_gaps(
    df: pd.DataFrame,
    temporal_res: dt.timedelta = dt.timedelta(hours=1)
) -> pd.DataFrame:
    """Fills NA values where there are data gaps for a consistently spaced
    time series.
    """
    if min(df.shape) > 0:
        t = np.array(df["time"], dtype="datetime64[ns]")
        t_full = pd.DataFrame({'tfull': np.arange(
            min(t), max(t) + np.timedelta64(temporal_res),
            np.timedelta64(temporal_res)
        )})
        df_corrected = pd.merge(
            left=t_full, right=df, left_on="tfull", right_on="time",
            how="left").drop("time", axis=1).rename({"tfull": "time"}, axis=1
                                                    )
        # Fill in X and Y for the missing values
        df_corrected.fillna(value={
            'x': df_corrected['x'][0],
            'y': df_corrected['y'][0],
        }, inplace=True)
        return df_corrected
    else:
        return df


def get_continuous_times(
    df: pd.DataFrame,
    temporal_res: dt.timedelta = dt.timedelta(hours=1),
    max_nans: int = 30,
) -> list[pd.DataFrame]:
    """Return a list of continuous times for seasonally split dataset.

    Args:
        df (pd.DataFrame):
            dataframe with a "time" column
        temporal_res (dt.timedelta):
            temporal resolution of the dataset.
        max_nans (int):
            maximum allowable NaN values in a row to still be considered
            continuous.

    Returns:
        list[pd.DataFrame]:
            list of dataframes with continuous times (likely yearly)
    """
    t = np.array(df["time"], dtype="datetime64[ns]")
    delta_t = np.diff(t).astype('timedelta64[s]')  # nanoseconds to seconds
    thresh = temporal_res * max_nans
    time_diff_ind = list(np.where(delta_t > thresh)[0])
    ends = time_diff_ind.copy()
    ends.append(df.shape[0])
    starts = [1] + [x + 1 for x in ends[:-1]]
    df_list = [correct_gaps(df[s:e], temporal_res)
               for s, e in zip(starts, ends)]
    return ([x for x in df_list if min(x.shape) > 0])


def cosmos_fit_acs(
    ntr_spatial: xr.Dataset,
    ts: dt.datetime,
    lagmax: int,
    dist_acf: str,
    years: range,
    n_days: int
) -> dict[str, float]:
    """Fit CoSMoS (marginal distribution, ACS function) for one window.
    Adapted from PyCoSMoS.
    NOTE: This does not stratify data seasonally.

    Args:
        ntr_spatial(xr.Dataset | pd.DataFrame):
            Dataset with each variable clipped to the spatial window
        ts (dt.datetime):
            temporal center of moving window
        lagmax(int): 
            max autocorrelation lag to compute
        dist_acf (str):
            ACF distribution function, options: 'paretoII', 'weibull', 'burrXII'
        years(range[int]):
            for MWM, range of years to use
        n_days(int):
            for MWM, days in moving window

    Returns:
        tuple[float, list, dict, list]:
            dict:
                par_acs (the ACS function parameters)
            list:
                acs_parametric (the parametric autocorrelation values for 1:lags)
    """
    ntr_window = (
        mwm.make_temporal_moving_window(
            ntr_spatial, years, ts.month, ts.day, ts.hour, n_days
        )
        .dropna(dim="time")
        .to_dataframe()['ntr']
    )
    if not ntr_window.empty:
        # Group by location
        groups = ntr_window.reset_index().groupby(['x', 'y'])
        dfs = [ntr_window.iloc[inds]
               .reset_index()
               for inds in groups.groups.values()]
        # Get a list of continuous dfs
        df_list = utils.flatten_list(
            [get_continuous_times(cont_df, dt.timedelta(hours=1), lagmax)
                for cont_df in dfs]
        )
        eacs = discontinuous_acs(df_list=df_list, lag_max=lagmax)
        eacs = pd.DataFrame({
            'lag': range(1, len(eacs) + 1),
            'rho': eacs
        })
        if not eacs.empty:
            params = fit_acs(dist_acf, eacs['lag'].max(), eacs)
            return pd.DataFrame(
                [{'ts': ts, 'param': k, 'value': v} for k, v in params.items()]
            )


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs---------------------------------
    upsampled_ocean_dir = snakemake.params['upsampled_ocean_dir']
    glob_pattern = snakemake.params['glob_pattern']
    startyear = int(snakemake.params['sy'])
    endyear = int(snakemake.params['ey'])
    ref_grid = xr.open_dataset(snakemake.params['ref_grid'])
    acs_df_csv_fname = snakemake.output[0]
    ndays = int(snakemake.params['ndays'])
    ngrids = int(snakemake.params['ngrids'])
    xmin = float(snakemake.params['xmin'])
    ymin = float(snakemake.params['ymin'])
    xmax = float(snakemake.params['xmax'])
    ymax = float(snakemake.params['ymax'])
    acsfamily = snakemake.params['acsfamily']
    lagmax = snakemake.params['lagmax']
    lat = float(snakemake.params['lat'])
    lon = float(snakemake.params['lon'])
    interval = dt.timedelta(days=1)
    # -Script------------------------------------------------------------
    files = [
        f for f in upsampled_ocean_dir.glob(glob_pattern)
        if (startyear <= int(str(f)[-7:-3]) <= endyear)
        and (os.stat(f).st_size > 0)
    ]
    grid = ref_grid.sel(x=slice(xmin, xmax),
                        y=slice(ymin, ymax))
    years = range(startyear, endyear)
    ntr_spatial_list = [
        ds['ntr'] for f in files
        if ((ds := mwm.make_spatial_moving_window(
            xr.open_dataset(f),
            lat, lon, grid, ngrids,
            apply_ocean_mask=False
        )).sizes['y'] > 0) and (ds.sizes['x'] > 0)
    ]
    if len(ntr_spatial_list) == 0:
        Path(acs_df_csv_fname).touch()
    else:
        ntr_spatial = xr.concat(ntr_spatial_list, dim="time")
        if ntr_spatial.dropna(dim="time").to_dataframe().empty:
            Path(acs_df_csv_fname).touch()
        else:
            # loop over daily time steps ---------------------------------------
            acs_dfs = [
                cosmos_fit_acs(
                    ntr_spatial, ts, lagmax, acsfamily, years, ndays
                ) for ts in utils.time_steps(interval)
            ]
            # Combine dataframes and write to csv ------------------------------
            acs = [x for x in acs_dfs if x is not None]
            if len(acs) > 0:
                acs_df = pd.concat(acs)
                acs_df.to_csv(acs_df_csv_fname)


if __name__ == "__main__":
    main()
