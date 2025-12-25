
"""Rename the ERA5 dimensions to (x, y, t) and resample hourly to daily.
"""

from pathlib import Path
import xarray as xr
import pandas as pd
import datetime as dt
from snakemake.script import snakemake


def rename_era5(ds: xr.Dataset) -> xr.Dataset:
    """Renames:
        longitude -> x
        latitude -> y
        valid_time -> time
    and drops number and expver coordinates for one netCDF file.
    """
    ds = ds.rename(
        {'longitude': "x", 'latitude': "y"})
    if "valid_time" in ds.coords:
        ds = ds.rename({'valid_time': "time"})
    if "number" in ds.coords:
        ds = ds.drop_vars("number")
    if "expver" in ds.coords:
        ds = ds.drop_vars("expver")
    return ds


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    year = snakemake.params['year']
    fname_out_daily = snakemake.output['daily']
    rawdata_dir = snakemake.params['rawdata_dir']
    #-Script------------------------------------------------------------
    fnames_in_raw = [
        rawdata_dir.joinpath(Path(f).name) for f in
        pd.read_csv(
            Path(snakemake.input['available_files'])
        )['era5_files']
        .tolist()
        if f[-10:-6] == year
    ]
    ds_year = (
        xr.open_mfdataset(
            fnames_in_raw,
            preprocess=rename_era5
        ).resample(
            {"time": dt.timedelta(days=1)}
        ).mean()
    )
    ds_year.to_netcdf(fname_out_daily)


if __name__ == "__main__":
    main()
