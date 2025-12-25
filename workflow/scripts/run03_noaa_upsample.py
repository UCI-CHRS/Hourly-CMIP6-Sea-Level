
"""Upsample NOAA tide, MSL, NTR, and TWL datasets to one of the CMIP6 grids."""

import datetime as dt
import glob
import os
import re
from pathlib import Path
import pandas as pd
import xarray as xr
from snakemake.script import snakemake
from aux import geo


def upsample(
    ds_match: xr.Dataset,
    df: pd.DataFrame,
    xvar: str = "lon",
    yvar: str = "lat",
    point_df_vars: list[str] = ['msl', 'twl', 'ntr', 'tides'],
) -> None:
    """Upsample one NOAA file."""
    if all([x > 0 for x in df.shape]):  # don't do empty dataframes
        gridded = geo.point_to_grid(
            ds_match,
            df,
            point_df_vars=point_df_vars,
            csv_time_var="time",
            point_df_xvar=xvar,
            point_df_yvar=yvar,
            x="x",
            y="y",
            crs="EPSG:4326",
        )
        xds = geo.cell_geometry_to_xarray(gridded)
        return xds


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    gridname = snakemake.params['gridname']
    year = snakemake.params['year']
    endyear = snakemake.params['endyear']
    ref_grid = Path(snakemake.input['ref_grid'])
    hourly_in = snakemake.input['hourly']
    daily_in = snakemake.input['daily']
    monthly_in = snakemake.input['monthly']
    hourly_out = snakemake.output['upsampled_hourly']
    daily_out = snakemake.output['upsampled_daily']
    monthly_out = snakemake.output['upsampled_monthly']
    #-Script------------------------------------------------------------
    ref_cmip_grid_ds = xr.open_dataset(ref_grid)
    df_hourly = pd.read_csv(hourly_in)
    df_daily = pd.read_csv(daily_in)
    if int(year) <= int(endyear):
        df_monthly = pd.read_csv(monthly_in)
        df_monthly['time'] = [y + m/12 for y,
                            m in zip(df_monthly['Year'], df_monthly['Month'])]
        hourly = upsample(
            ds_match=ref_cmip_grid_ds,
            df=df_hourly,
            xvar="lng",
            yvar="lat",
        )
        daily = upsample(
            ds_match=ref_cmip_grid_ds,
            df=df_daily,
            xvar="lng",
            yvar="lat",
        )
        monthly = upsample(
            ds_match=ref_cmip_grid_ds,
            df=df_monthly,
            xvar="lng",
            yvar="lat",
            point_df_vars=['msl']
        ).rename({'time': 'years'})
        hourly.to_netcdf(hourly_out)
        daily.to_netcdf(daily_out)
        monthly.to_netcdf(monthly_out)
    else:  # future tides only
        hourly = upsample(
            ds_match=ref_cmip_grid_ds,
            df=df_hourly,
            xvar="lng",
            yvar="lat",
            point_df_vars=['tides']
        )
        daily = upsample(
            ds_match=ref_cmip_grid_ds,
            df=df_daily,
            xvar="lng",
            yvar="lat",
            point_df_vars=['tides']
        )
        hourly.to_netcdf(hourly_out)
        daily.to_netcdf(daily_out)
        Path(monthly_out).touch()


if __name__ == "__main__":
    main()
