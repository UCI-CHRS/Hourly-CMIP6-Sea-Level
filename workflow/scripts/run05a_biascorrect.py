
"""Get the mean and standard deviation of the u- and v- wind components
and sea level pressure variables.
"""

import os
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
from snakemake.script import snakemake


def getstats(ds: xr.Dataset, time_window: int, spatial_window: int):
    """
    Get the mean and standard deviation within each 
    spatiotemporal rolling window. 
    """
    # Temporal rolling over TIME
    rolled_t = ds.rolling(
        time=time_window,
        center=True,
        min_periods=1
    ).construct("twindow")
    # Add spatial rolling - x dimension
    rolled_tx = rolled_t.rolling(
        x=spatial_window,
        center=True,
        min_periods=1
    ).construct("xwindow")
    # Add spatial rolling - y dimension
    rolled_txy = rolled_tx.rolling(
        y=spatial_window,
        center=True,
        min_periods=1
    ).construct("ywindow")
    # Stack dimensions
    # Add day-of-year coordinate
    dayofyear = ds.time.dt.dayofyear.data
    rtxy = rolled_txy.assign_coords(dayofyear=("time", dayofyear))
    # Now group by day-of-year of the central time to pool across years
    stacked = rtxy.stack(window=("twindow", "xwindow", "ywindow"))
    mean = stacked.groupby("dayofyear").mean(
        dim=("window", "time"), skipna=True)
    std = stacked.groupby("dayofyear").std(dim=("window", "time"), skipna=True)
    # Final summary dataset
    summary = xr.Dataset(
        {f"{v}_mean": mean[v] for v in ds.data_vars} |
        {f"{v}_std":  std[v] for v in ds.data_vars}
    )
    return summary


def main(
    upsampled_atmos_daily: str,
    glob_pattern_daily_atmos: str,
    cmip: str,
    ndays: int,
    ngrids: int,
    startyear: int,
    endyear: int,
    fname_out: str
):
    """Main script to run via snakemake."""
    # Moving window
    atmos_files = [
        f for f in upsampled_atmos_daily.glob(glob_pattern_daily_atmos)
        if (startyear <= int(str(f)[-7:-3]) <= endyear)
        and (os.stat(f).st_size > 0)
    ]
    ds_atmos = (
        xr.open_mfdataset(atmos_files)
        .sortby('time')
        .compute()
        .rename({'u10': 'uas', 'v10': 'vas', 'msl': 'psl'})
    )
    ds_cmip = (
        xr.open_dataset(cmip)
        .pipe(lambda ds: ds.assign_coords(time=ds.time.dt.floor("D")))
        .sel(time=slice(f"{startyear}-01-01", f"{endyear}-12-31"))
        .drop_vars('tos')
    )
    hist = getstats(ds_cmip, ndays+1, ngrids).expand_dims(stat_source=["hist"])
    obs = getstats(ds_atmos, ndays+1, ngrids).expand_dims(stat_source=["obs"])
    # Combine
    combined = xr.concat([hist, obs], dim="stat_source")
    data = combined.to_array("variable")
    # Write out
    data.to_dataset(dim="variable").to_netcdf(fname_out)


def parse_args():
    args = dict(
        # -Snakemake params, inputs, outputs---------------------------------
        upsampled_atmos_daily=snakemake.params['upsampled_atmos_daily'],
        glob_pattern_daily_atmos=snakemake.params['glob_pattern_daily_atmos'],
        cmip=snakemake.input['cmip'],
        ndays=int(snakemake.params['ndays']),
        ngrids=int(snakemake.params['ngrids']),
        startyear=int(snakemake.params['sy']),
        endyear=int(snakemake.params['ey']),
        fname_out=snakemake.output[0]
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
