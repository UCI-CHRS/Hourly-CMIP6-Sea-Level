
"""Daily and monthly NTR regression fits"""

import datetime as dt
from pathlib import Path
import os
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from snakemake.script import snakemake
from aux import mwm, metrics, utils, enso


def standardize(x):
    """Returns a standardized Z-score of x to use with daily NTR regression.
    Also returns min and max values to identify extrapolation when applied to CMIP.
    """
    xmin = np.min(x)
    xmax = np.max(x)
    xmean = np.mean(x)
    xstd = np.std(x)
    xnorm = (x - xmean)/xstd
    return xnorm, xmean, xstd, xmin, xmax


def regress_daily_ntr(
    uwnd: pd.Series,
    vwnd: pd.Series,
    slp: pd.Series,
    ntr: pd.Series
) -> dict[str, float]:
    """Daily mean non-tidal residual estimated by linear regression on wind stress
        and sea level pressure. Mean and std dev values used for standardization
        are reported to check for extrapolation when applying to CMIP data.

    Args:
        uwnd(pd.Series):
            eastward/zonal wind speed data
        vwnd(pd.Series):
            northward/meridional wind speed data
        slp(pd.Series):
            sea level pressure data
        ntr(pd.Series):
            non-tidal residual(storm surge) data

    Returns:
        dict[str, float]:
            dictionary of regression parameters
    """
    # Convert wind speed to wind pseudo-stress
    tauu = np.multiply(np.multiply(np.sign(uwnd), uwnd), uwnd)
    tauv = np.multiply(np.multiply(np.sign(vwnd), vwnd), vwnd)
    # Standardize variables with Z-score
    tauu_norm, tauu_mean, tauu_std, tauu_min, tauu_max = standardize(tauu)
    tauv_norm, tauv_mean, tauv_std, tauv_min, tauv_max = standardize(tauv)
    slp_norm, slp_mean, slp_std, slp_min, slp_max = standardize(slp)
    # Concatenate data pairs into matrix
    X = np.concatenate(
        [np.expand_dims(tauu_norm, axis=1),
         np.expand_dims(tauv_norm, axis=1),
         np.expand_dims(slp_norm, axis=1),
         np.expand_dims(ntr, axis=1)
         ], axis=1
    )
    if X.shape[0] > 1:
        # linear regression
        reg = LinearRegression().fit(X[:, :3], X[:, -1])
        r2 = reg.score(X[:, :3], X[:, -1])
        m, b = reg.coef_, reg.intercept_
        out = {
            "m_tauu": m[0],
            "m_tauv": m[1],
            "m_slp": m[2],
            "b": b,
            "r2": r2,
            "z_mean_tauu": tauu_mean,
            "z_std_tauu": tauu_std,
            "min_tauu": tauu_min,
            "max_tauu": tauu_max,
            "z_mean_tauv": tauv_mean,
            "z_std_tauv": tauv_std,
            "min_tauv": tauv_min,
            "max_tauv": tauv_max,
            "z_mean_slp": slp_mean,
            "z_std_slp": slp_std,
            "min_slp": slp_min,
            "max_slp": slp_max,
        }
        yfit = reg.predict(X[:, :3])
        yobs = X[:, -1]
        # add and rename error metrics
        val = metrics.regression_metrics(X[:, :3], yobs, yfit)
        val['fstat_tauu'] = val.pop('f_stat_1')
        val['fstat_tauu_pval'] = val.pop('f_stat_p_val_1')
        val['fstat_tauv'] = val.pop('f_stat_2')
        val['fstat_tauv_pval'] = val.pop('f_stat_p_val_2')
        val['fstat_slp'] = val.pop('f_stat_3')
        val['fstat_slp_pval'] = val.pop('f_stat_p_val_3')
        return out | val


def regress_monthly_ntr(
    nino_3_4_sst: pd.Series,
    monthly_ntr: pd.Series
) -> dict[str, float]:
    """Monthly regression: monthly NTR on sst
        NINO3.4 reference:
        https: // climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and -tni
        Get nino3.4 sst anomalies

    Args:
        nino_3_4_sst(pd.Series with month, year MultiIndex):
            NINO3.4 SST data(get from read_enso())
        monthly_ntr(pd.Series with month, year MultiIndex):
            Monthly non-tidal residual data

    Returns:
        dict[str, float]:
            Dictionary of regression parameters for msl ~ nino34
    """
    # linear regression
    X = nino_3_4_sst.to_numpy().reshape(-1, 1)
    Y = monthly_ntr.to_numpy().reshape(-1, 1)
    if (len(X) > 1) and (len(Y) > 1):
        reg = LinearRegression().fit(X, Y)
        m, b = reg.coef_, reg.intercept_
        r2 = reg.score(X, Y)
        # Add standard deviation for 1991-2020 for rescaling the CMIP data
        # and min and max NINO 3.4 values used for regression fit
        out = {
            'm': m[0][0],
            'b': b[0],
            'r2': r2,
            'stdev': np.std(nino_3_4_sst),
            'min_nino34': np.min(X),
            'max_nino34': np.max(X),
        }
        yfit = reg.predict(X)
        # add error metrics
        val = metrics.regression_metrics(X, Y, yfit)
        return out | val


def regression_one_window(
    ts: dt.datetime,
    years: range,
    n_days: int,
    ntr_spatial_daily: pd.DataFrame,
    atmos_spatial_daily: pd.DataFrame,
    nino_3_4_sst: pd.DataFrame
) -> None:
    """Helper function to run regression fits for one spatiotemporal window.

    Args:
        ts(dt.datetime):
            time step in middle of window
        lat(float):
            latitude in middle of window
        lon(float):
            longitude in middle of window
        years(range):
            range of start to end year
        temporal_res(dt.timedelta):
            temporal resolution of the temporally downscaled data
        grid(xr.Dataset):
            dataset with the CMIP grid on which the window size is based
        n_grids(int):
            number of CMIP grids along one dimension of the window
        n_days(int):
            number of days in the moving window
        ds_spatial(xr.Dataset):
            dataset clipped to the spatial moving window
        ntr_spatial_daily(pd.DataFrame):
            Daily NTR dataframe clipped to the spatial window
        ntr_spatial_monthly(pd.DataFrame):
            Monthly NTR dataframe clipped to the spatial window

    Returns:
        None (results saved directly to disk)
    """
    ntr_daily_window = mwm.make_temporal_moving_window(
        ntr_spatial_daily, years=years, month=ts.month, day=ts.day,
        hour=ts.hour, n_days=n_days
    ).to_dataframe().reset_index()
    atmos_daily_window = mwm.make_temporal_moving_window(
        atmos_spatial_daily, years=years, month=ts.month, day=ts.day,
        hour=ts.hour, n_days=n_days
    ).to_dataframe()
    # Decompose NTR into long-term and short-term components
    ntr_lf, ntr_hf = enso.decomposentr(ntr_daily_window, n_days)
    sst_daily = enso.interpolate_sst(nino_3_4_sst).set_index('time')
    sst_window = mwm.make_temporal_moving_window(
        sst_daily, years, ts.month, ts.day, ts.hour, n_days
    )
    # Daily regression
    daily_df = ntr_hf.set_index(['x', 'y', 'time']).join(
        atmos_daily_window).dropna()
    daily_params = regress_daily_ntr(
        daily_df['u10'],
        daily_df['v10'],
        daily_df['msl'],
        daily_df['ntr']
    )
    # Monthly regression
    monthly_df = (
        sst_window
        .join(ntr_lf.drop(
            ['x', 'y'], axis=1)
            .set_index('time'))
        .dropna()
    )
    monthly_params = regress_monthly_ntr(
        monthly_df['nino34'], monthly_df['ntr'])
    if (monthly_params is not None) and (daily_params is not None):
        daily = pd.DataFrame(
            [{'ts': ts, 'param': k, 'value': v}
                for k, v in daily_params.items()]
        )
        monthly = pd.DataFrame(
            [{'ts': ts, 'param': k, 'value': v}
                for k, v in monthly_params.items()]
        )
        return {'daily': daily, 'monthly': monthly}


def combine_datasets(ocean_files, atmos_files, lat, lon, grid, ngrids):
    """Get spatial windows."""
    any_empty = False
    ntr_spatial_daily_list = [
        ds['ntr'] for f in ocean_files
        if ((ds := mwm.make_spatial_moving_window(
            xr.open_dataset(f),
            lat, lon, grid, ngrids,
            apply_ocean_mask=False
        )).sizes['y'] > 0) and (ds.sizes['x'] > 0)
    ]
    if len(ntr_spatial_daily_list) > 0:
        ntr_spatial_daily = xr.concat(ntr_spatial_daily_list, dim="time")
        ntr_spatial_daily['time'] = pd.to_datetime(ntr_spatial_daily.time)
    else:
        ntr_spatial_daily = None
        any_empty = True
    atmos_spatial_daily_list = [
        ds for f in atmos_files
        if (ds := mwm.make_spatial_moving_window(
            xr.open_dataset(f),
            lat, lon, grid, ngrids,
            apply_ocean_mask=False
        )).sizes['y'] > 0
    ]
    if len(atmos_spatial_daily_list) > 0:
        atmos_spatial_daily = xr.concat(atmos_spatial_daily_list, dim="time")
    else:
        any_empty = True
        atmos_spatial_daily = None
    if not any_empty:
        any_empty = (
            ntr_spatial_daily.dropna(dim="time").to_dataframe().empty or
            atmos_spatial_daily.dropna(dim="time").to_dataframe().empty
        )
    return (
        ntr_spatial_daily,
        atmos_spatial_daily,
        any_empty
    )


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs---------------------------------
    upsampled_ocean_daily = snakemake.params['upsampled_ocean_daily']
    glob_pattern_daily_ocean = snakemake.params['glob_pattern_daily_ocean']
    upsampled_atmos_daily = snakemake.params['upsampled_atmos_daily']
    glob_pattern_atmos_ocean = snakemake.params['glob_pattern_daily_atmos']
    ref_grid = xr.open_dataset(snakemake.params['ref_grid'])
    regression_daily_csv_fname = snakemake.output['daily']
    regression_monthly_csv_fname = snakemake.output['monthly']
    ndays = int(snakemake.params['ndays'])
    ngrids = int(snakemake.params['ngrids'])
    xmin = float(snakemake.params['xmin'])
    ymin = float(snakemake.params['ymin'])
    xmax = float(snakemake.params['xmax'])
    ymax = float(snakemake.params['ymax'])
    startyear = int(snakemake.params['sy'])
    endyear = int(snakemake.params['ey'])
    lat = float(snakemake.params['lat'])
    lon = float(snakemake.params['lon'])
    # -Script------------------------------------------------------------
    # Read non-tidal residual data
    interval = dt.timedelta(days=1)
    # clip extent of the CMIP grids
    grid = ref_grid.sel(x=slice(xmin, xmax),
                        y=slice(ymin, ymax))
    years = range(startyear, endyear)
    # Moving window
    ocean_files = [
        f for f in upsampled_ocean_daily.glob(glob_pattern_daily_ocean)
        if (startyear <= int(str(f)[-7:-3]) <= endyear)
        and (os.stat(f).st_size > 0)
    ]
    atmos_files = [
        f for f in upsampled_atmos_daily.glob(glob_pattern_atmos_ocean)
        if (startyear <= int(str(f)[-7:-3]) <= endyear)
        and (os.stat(f).st_size > 0)
    ]
    nino34 = pd.read_csv(snakemake.input['enso'])
    if (len(ocean_files) > 0) and (len(atmos_files) > 0):
        ntr_spatial_daily, atmos_spatial_daily, any_empty = combine_datasets(
            ocean_files, atmos_files, lat, lon, grid, ngrids
        )
        if any_empty:
            Path(regression_daily_csv_fname).touch()
            Path(regression_monthly_csv_fname).touch()
        else:
            # loop over daily time steps
            regression_daily_monthly_dfs = [
                regression_one_window(
                    ts,
                    years,
                    ndays,
                    ntr_spatial_daily,
                    atmos_spatial_daily,
                    nino34
                )
                for ts in utils.time_steps(interval)
            ]
            # Combine dataframes and write to csv
            regression_daily = [
                x['daily'] for x in regression_daily_monthly_dfs
                if x is not None
            ]
            if len(regression_daily) > 0:
                regression_daily_df = pd.concat(regression_daily)
                regression_daily_df.to_csv(regression_daily_csv_fname)
            regression_monthly = [
                x['monthly'] for x in regression_daily_monthly_dfs
                if x is not None
            ]
            if len(regression_monthly) > 0:
                regression_monthly_df = pd.concat(regression_monthly)
                regression_monthly_df.to_csv(regression_monthly_csv_fname)


if __name__ == "__main__":
    main()
