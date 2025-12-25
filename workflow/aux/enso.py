
import pandas as pd
import scipy
import xarray as xr
import numpy as np


def decomposentr(
    ntr_daily: pd.DataFrame,
    ndays: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        ntr_daily: Daily NTR data with x, y, time coordinates.
        ndays:
            This should be the same as the number of days in the moving window.

    Returns:
        ntr_lf: low frequency component used for ENSO regression
        ntr_hf: high frequency component used for wind and pressure regression
    """
    # N-day rolling mean:
    ntr_lf = (ntr_daily.groupby(['x', 'y'])
              .apply(lambda df: df.set_index("time"))['ntr']
              .rolling(ndays).mean().reset_index()
              )
    combined = ntr_daily.set_index(["x", "y", "time"]).join(
        ntr_lf.set_index(["x", "y", "time"]).rename({"ntr": "ntr_lf"}, axis=1))
    combined['ntr_hf'] = combined['ntr'] - combined['ntr_lf']
    combined.reset_index(inplace=True)
    return (combined[['time', 'x', 'y', 'ntr_lf']].rename({'ntr_lf': 'ntr'}, axis=1),
            combined[['time', 'x', 'y', 'ntr_hf']].rename({'ntr_hf': 'ntr'}, axis=1))


def detrended_enso_34_sst(
    obs_enso: pd.DataFrame,
    sst: xr.Dataset
) -> pd.Series:
    """
    Get the detrended ENSO 3.4 SST anomaly averaged for 120 - 170W, 5S - 5N
    for a gridded CMIP6 dataset.
    NOTE: The climatology period for the observed ENSO is 1991 to 2020,
    but here we will use the full available (detrended) period to get
    mu and std.

    Args:
        sst (xr.Dataset):
            Gridded sea surface temperature data

    Returns:
        pd.Series:
            Detrended ENSO 3.4 SST anomaly averaged for the
            120 - 170W, 5S - 5N region of the gridded sst dataset
    """
    # Get monthly mean values averaged for 120 - 170W, 5S - 5N
    # and resample to ~ the middle of the month
    min_lat, max_lat, min_lon, max_lon = -5, 5, -170, -120
    ds = (
        sst.sel(
            y=slice(min_lat, max_lat),
            x=slice(min_lon, max_lon)
        ).mean(dim=["x", "y"])
         .resample(time="MS")
         .mean()
    )
    ds = ds.assign_coords(time=ds.time + pd.Timedelta(days=15))
    # Convert to dataframe
    df = ds.to_dataframe()['tos']
    # Detrend time series
    _ = scipy.signal.detrend(df, type="linear", overwrite_data=True)
    # Remove climatological mean and
    # smooth anomalies with a 5-month rolling window (per NOAA/NCAR docs)
    df = (df - df.mean()).rolling(5, center=False).mean().dropna()
    # Rescale standard deviation to match that of 1991-2020
    sig_x = np.std(obs_enso)
    sig_y = np.std(df)
    sig_x_sig_y_ratio = sig_x/sig_y
    df = df * sig_x_sig_y_ratio
    # Set index to month-year
    df.index = pd.MultiIndex.from_arrays(
        [df.index.year, df.index.month],
        names=("year", "month")
    )
    return df


def interpolate_sst(sst: pd.DataFrame, sst_col="nino34") -> pd.DataFrame:
    """Transform monthly SST into daily data using cubic spline interpolation

    Args:
        sst (pd.DataFrame):
            DF of SST values, with year, month multiindex.

    Returns:
        pd.DataFrame:
            SST interpolated daily
    """
    df = sst.reset_index()
    # Set values to the 15th (middle) of the month
    times = [np.datetime64(f"{year}-{str(month).zfill(2)}-15")
             for year, month in zip(df.year, df.month)]
    times_d = [d.astype('d') for d in times]
    # Interpolate with a cubic spline
    cubic_spline = scipy.interpolate.CubicSpline(times_d, sst[sst_col].values)
    daily_d = np.arange(min(times_d), max(times_d))
    daily_sst = cubic_spline(daily_d)
    # return pd.Series(daily_sst, index = np.arange(min(times), max(times)))
    return pd.DataFrame({'time': np.arange(min(times), max(times)), sst_col: daily_sst})
