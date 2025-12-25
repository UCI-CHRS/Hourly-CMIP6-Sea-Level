"""Check for missing ERA5 data. """

from pathlib import Path
import os
import pandas as pd
import xarray as xr
from snakemake.script import snakemake


def check_corrupt(fname: Path) -> xr.Dataset | None:
    """Check if the files are readable. Remove if not.

    Args:
        fname (Path): file name to read

    Returns:
        xr.Dataset | None:
            dataset if readable, None if not.
    """
    try:
        ds = xr.open_dataset(
            fname, engine="netcdf4", decode_cf=False,
            decode_times=False, decode_coords=False,
            use_cftime=False, decode_timedelta=False
        )
        return fname
    except (OSError, ValueError):
        return None


def sum_missing_over_time(
    ds: xr.Dataset
) -> pd.DataFrame:
    """Sum missing values, drop non-dim coords, and convert to dataframe.
    If missing values are not NaN (e.g., -9999),
    they need to be converted first.

    Args:
        ds (xr.Dataset): dataset with x, y (spatial) and time dimensions

    Returns:
        xr.Dataset: The original dataset, except the time dimension is
            flattened and replaced with the number of missing values
            over time.
    """
    dimnames = list(ds.dims)
    # sometimes its "valid_time"
    tdim = [x for x in dimnames if "time" in x][0]
    ds_missing = ds.isnull().sum(dim=tdim)
    df_missing = ds_missing.drop_vars(
        list(set(ds_missing.coords).difference(ds_missing.dims))
    ).to_dataframe().reset_index()
    return df_missing


def preprocess(ds):
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'expver' in ds.coords:
        ds = ds.drop_vars("expver")
    return ds


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    atmos_nc_dir = snakemake.params['atmos_nc_dir']
    fnames = snakemake.params['fnames']
    missing = snakemake.output['missing']
    available = snakemake.output['available']
    #-Script------------------------------------------------------------
    # Check for corrupt files
    working_paths = [check_corrupt(atmos_nc_dir.joinpath(f)) for f in fnames]
    if any(x is not None for x in working_paths):
        # Sum missing values over time
        ds_year = xr.open_mfdataset(
            working_paths,
            preprocess=preprocess
        )
        df_missing = sum_missing_over_time(ds_year)
    else:
        df_missing = pd.DataFrame(
            columns=['latitude',  'longitude',  'u10',  'v10', 'msl']
        )
    df_missing.to_csv(missing, index=False)
    # Report available files
    available_files = pd.DataFrame({
        'era5_files': [str(f) for f in working_paths]
    })
    available_files.to_csv(available, index=False)


if __name__ == "__main__":
    main()
