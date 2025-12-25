
from pathlib import Path
import os
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from snakemake.script import snakemake


def init_nc(df, lons, lats):
    lats = list(set(lats))
    lons = list(set(lons))
    datavars = list(df.columns)
    time_axis = df.index
    return xr.Dataset(
        data_vars={
            'ntr': (
                ['x', 'y', 'time'],
                np.full([len(lons), len(lats), len(time_axis)], np.nan)
            )
        },
        coords={'x': lons, 'y': lats, 'time': time_axis}
    )


def combine_csvs_into_nc(csvs):
    # Get dimensions
    latlon_files = [Path(f) for f in csvs if os.stat(f).st_size > 0]
    latlons = list({
        (float(str(f.name).split("_")[3]),  # lat
            float(str(f.name).split("_")[5]))  # lon
        for f in latlon_files
    })
    # Get unique dimensions for initializing the dataset
    lats = list({x[0] for x in latlons})
    lons = list({x[1] for x in latlons})
    INITIALIZE = True
    # Create xarray dataset
    if len(latlon_files) > 0:
        for f in latlon_files:
            lat = float(str(f.name).split("_")[3])
            lon = float(str(f.name).split("_")[5])
            # Read csvs for current lat/lon
            df = pd.read_csv(f, index_col=0, parse_dates=True)
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
    return nc


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs--------------------------------
    marg = snakemake.wildcards['marg']
    acs = snakemake.wildcards['acs']
    ntr_csv_dir = snakemake.params['ntr_csv_dir']
    nc_out = snakemake.output[0]
    # -Script-----------------------------------------------------------
    # NTR
    ntr_hourly_csvs = list(Path(ntr_csv_dir).glob(
        f"ntr_hourly_*_marg_{marg}_acs_{acs}.csv"))
    ntr_hourly = combine_csvs_into_nc(ntr_hourly_csvs)
    ntr_hourly = ntr_hourly.where(np.isfinite(ntr_hourly))
    ntr_hourly.attrs = {
        'ntr': (
            "Hourly non-tidal residual, estimated based on NINO3.4 index, "
            "zonal and meridional wind stress, and sea level pressure."
        )
    }
    ntr_hourly.to_netcdf(nc_out)


if __name__ == "__main__":
    main()
