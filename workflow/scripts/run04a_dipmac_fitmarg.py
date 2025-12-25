
"""Fit marginal distribution parameters for DiPMaC."""

import os
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from snakemake.script import snakemake
from aux import mwm, utils, stats as s


def cosmos_fit_marg(
    ts: dt.datetime,
    ntr_spatial: xr.Dataset | pd.DataFrame,
    years: range,
    n_days: int,
    marginal_family: str
) -> None:
    """Fit CoSMoS marginal distribution for one spatiotemporal window.
    Adapted from PyCoSMoS.
    NOTE: This does not stratify data seasonally.

    Args:
        ts (dt.datetime):
        ntr_spatial (xr.Dataset | pd.DataFrame):
            Dataset with each variable clipped to the spatiotemporal window
        years (range[int]): for MWM, range of years to use
        n_days (int): # for MWM, days in moving window
        marginal_family (str): marginal family name

    Returns:
        None (fits saved directly to disk)
    """
    # Fit dipmac to values in window
    ntr_window = (
        mwm.make_temporal_moving_window(
            ntr_spatial, years, ts.month, ts.day, ts.hour, n_days
        )
        .dropna(dim="time")
        .to_dataframe()['ntr']
    )
    if not ntr_window.empty:
        params = s.fit_distribution(
            marginal_family,
            pd.Series(ntr_window.values)
        )
        # Save results to dataframe with datetime index
        if params is not None:
            return pd.DataFrame(
                [{'ts': ts, 'param': k, 'value': v} for k, v in params.items()]
            )


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        upsampled_ocean_dir=snakemake.params['upsampled_ocean_dir'],
        glob_pattern=snakemake.params['glob_pattern'],
        ref_grid=xr.open_dataset(snakemake.params['ref_grid']),
        marg_df_csv_fname=snakemake.output[0],
        ndays=int(snakemake.params['ndays']),
        ngrids=int(snakemake.params['ngrids']),
        xmin=float(snakemake.params['xmin']),
        ymin=float(snakemake.params['ymin']),
        xmax=float(snakemake.params['xmax']),
        ymax=float(snakemake.params['ymax']),
        startyear=int(snakemake.params['sy']),
        endyear=int(snakemake.params['ey']),
        margfamily=snakemake.params['margfamily'],
        lat=float(snakemake.params['lat']),
        lon=float(snakemake.params['lon']),
    )
    return args


def main(
    upsampled_ocean_dir: str,
    glob_pattern: str,
    ref_grid: xr.Dataset,
    marg_df_csv_fname: str,
    ndays: int,
    ngrids: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    startyear: int,
    endyear: int,
    margfamily: str,
    lat: float,
    lon: float,
):
    """Main script to run via snakemake."""
    # Read non-tidal residual data
    interval = dt.timedelta(days=1)
    # clip extent of the CMIP grids
    grid = ref_grid.sel(x=slice(xmin, xmax),
                        y=slice(ymin, ymax))
    years = range(startyear, endyear)
    # Moving window
    files = [
        f for f in upsampled_ocean_dir.glob(glob_pattern)
        if (startyear <= int(str(f)[-7:-3]) <= endyear)
        and (os.stat(f).st_size > 0)
    ]
    ntr_spatial_list = [
        ds['ntr'] for f in files
        if ((ds := mwm.make_spatial_moving_window(
            xr.open_dataset(f),
            lat, lon, grid, ngrids,
            apply_ocean_mask=False
        )).sizes['y'] > 0) and (ds.sizes['x'] > 0)
    ]
    if len(ntr_spatial_list) == 0:
        Path(marg_df_csv_fname).touch()
    else:
        ntr_spatial = xr.concat(ntr_spatial_list, dim="time")
        if ntr_spatial.dropna(dim="time").to_dataframe().empty:
            Path(marg_df_csv_fname).touch()
        else:
            # loop over daily time steps
            marg_dfs = [
                cosmos_fit_marg(ts, ntr_spatial, years, ndays, margfamily)
                for ts in tqdm(utils.time_steps(interval))
            ]
            # Combine dataframes and write to csv
            marg = [x for x in marg_dfs if x is not None]
            if len(marg) > 0:
                marg_df = pd.concat(marg)
                marg_df.to_csv(marg_df_csv_fname)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
