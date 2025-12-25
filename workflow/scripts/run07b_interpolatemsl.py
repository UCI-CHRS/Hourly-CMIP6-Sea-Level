
from pathlib import Path
import os
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt
from snakemake.script import snakemake


def interp_along_time(ds: xr.Dataset, new_time):
    """
    Interpolate an xarray Dataset along time using
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
    interpolation.

    Args:
        ds : xr.Dataset
            Input data with a 'time' dimension.
        new_time : array-like of datetime64
            New time coordinate for interpolation.

    Returns:
        xr.Dataset
            Interpolated data on the new time grid.
    """

    def pchip_interpolate(y, x_old, x_new):
        # Handle NaNs
        if np.all(np.isnan(y)):
            return np.full_like(x_new, np.nan, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.full_like(x_new, np.nan, dtype=float)
        pchip = PchipInterpolator(x_old[mask], y[mask], axis=0)
        return pchip(x_new)

    t0, t1 = ds.time[0].values, ds.time[-1].values
    x_old = (ds.time.values - t0) / np.timedelta64(1, "D")
    x_old = x_old.astype(float)
    x_new = (new_time - t0) / np.timedelta64(1, "D")
    x_new = x_new.astype(float)

    vars_to_interp = [v for v in ds.data_vars if "time" in ds[v].dims]
    out = {}
    for v in vars_to_interp:
        da = ds[v]
        interp = xr.apply_ufunc(
            pchip_interpolate,
            da,
            x_old,
            x_new,
            input_core_dims=[['time'], ['time'], ['time_new']],
            output_core_dims=[['time_new']],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da.dtype],
        )
        interp = interp.rename({"time_new": "time"}
                               ).assign_coords(time=new_time)
        out[v] = interp
    # Keep non-time coords from original
    return (
        xr.Dataset(
            out,
            coords={k: v for k, v in ds.coords.items()
                    if k != "time"}
        ).assign_coords(time=new_time)
    )


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs--------------------------------
    msl_monthly_file = snakemake.input['msl_monthly']
    ntr_file = snakemake.input['ntr']
    nc_out = snakemake.output['msl_hourly']
    # -Script-----------------------------------------------------------
    # Read NTR to get coordinates
    ntr = xr.open_dataset(ntr_file, chunks={'time': 100})
    x_vals, y_vals, t_vals = ntr.x, ntr.y, ntr.time
    # MSL
    msl_monthly = xr.open_dataset(msl_monthly_file).sel(
        x=x_vals, y=y_vals)
    msl_monthly = (
        msl_monthly[['ice_melt', 'msl']]
        .sel(time=slice(t_vals.min().values, t_vals.max().values))
        .sortby("time")
    )
    hourly_times = pd.date_range(
        msl_monthly.time.min().values,
        msl_monthly.time.max().values,
        freq="h"
    )
    msl_hourly = interp_along_time(
        msl_monthly, hourly_times.values
    )
    # Check that data falls within range. Fall back to linear interpolation
    # if not.
    tol = 1e-2
    if (
        (msl_hourly.msl.min().values < msl_monthly.msl.min().values - tol) or 
        (msl_hourly.msl.max().values > msl_monthly.msl.max().values + tol)
    ):
        msl_hourly = msl_monthly.interp(time=hourly_times.values)
    common_times = np.intersect1d(
        ntr['time'].values,
        msl_hourly['time'].values
    )
    msl_hourly = msl_hourly.sel(x=x_vals, y=y_vals, time=common_times)
    msl_hourly.attrs = {
        'ice_melt': "Global monthly ice melt (units: m)",
        'msl': "Sum of ice_melt, zostoga, and zos (units: m)",
    }
    msl_hourly.to_netcdf(nc_out)


if __name__ == "__main__":
    main()
