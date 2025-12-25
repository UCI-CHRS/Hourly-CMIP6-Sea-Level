"""Utility script for reading the individual moving window parameter CSVs
into .nc files.
"""

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from snakemake.script import snakemake


def init_nc(df, lons, lats):
    lats = list(set(lats))
    lons = list(set(lons))
    datavars = list(df.columns)
    time_axis = df.index
    return xr.Dataset(
        data_vars={
            var: (
                ['x', 'y', 'time'],
                np.full([len(lons), len(lats), len(time_axis)], np.nan)
            )
            for var in datavars
        },
        coords={'x': lons, 'y': lats, 'time': time_axis}
    )


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs---------------------------------
    nc_file = snakemake.output[0]
    input_files = snakemake.input
    # -Script------------------------------------------------------------
    # Get dimensions
    latlon_files = [Path(f) for f in input_files if os.stat(f).st_size > 0]
    if len([f for f in latlon_files if 'actpnts' in str(f)]) > 0:
        # Touch a dummy file for actpnts, which doesn't have a time axis.
        Path(nc_file).touch()
    else:
        latlons = list({
            (float(str(f.name).split("_")[1]),  # lat
             float(str(f.name).split("_")[3]))  # lon
            for f in latlon_files
        })
        # Get unique dimensions for initializing the dataset
        lats = list({x[0] for x in latlons})
        lons = list({x[1] for x in latlons})
        INITIALIZE = True
        # Create xarray dataset
        if len(latlon_files) > 0:
            for f in latlon_files:
                lat = float(str(f.name).split("_")[1])
                lon = float(str(f.name).split("_")[3])
                # Read csvs for current lat/lon
                df = pd.read_csv(f, index_col=1, parse_dates=True)
                # dtype={'param': str, 'value': float})
                df = df[['param', 'value']].pivot(columns='param')['value']
                df.index.name = "time"
                if INITIALIZE:
                    # Initialize xarray datasets
                    nc = init_nc(df, lons, lats).sortby(["x", "y"])
                    INITIALIZE = False
                # fill into the xarray dataset
                x_coord = nc.sel(x=lon, y=lat, method="nearest").x.values
                y_coord = nc.sel(x=lon, y=lat, method="nearest").y.values
                try:
                    nc.loc[dict(x=x_coord, y=y_coord)] = xr.Dataset(df)
                except ValueError as e:
                    warnings.warn(f"Encountered error {e} for file {f}")
            nc = nc.stack({'gridcenter': ('x', 'y')}).dropna(
                dim="gridcenter").unstack()
            nc.to_netcdf(nc_file)


if __name__ == "__main__":
    main()
