
"""Read raw CMIP files. """

from pathlib import Path
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
import xmip.preprocessing as xprep
from . import geo


def xmip_wrapper(ds_orig: xr.Dataset) -> xr.Dataset:
    """CMIP preprocessing function.
    Models from MRI or ISPL can be finicky.
    Applies all XMIP renaming/data cleaning to a CMIP6 dataset.
    See cmip6-preprocessing.readthedocs.io
    NOTE: this changes lat and lon coords to y and x

    Args:
        ds_orig (xr.Dataset):
            raw CMIP6 dataset

    Returns:
        xr.Dataset:
            processed dataset with x, y, time coordinates
    """
    ds = (ds_orig.copy()
          .pipe(xprep.rename_cmip6)
          .pipe(xprep.promote_empty_dims)
          .pipe(xprep.broadcast_lonlat)
          .pipe(xprep.replace_x_y_nominal_lat_lon)
          .pipe(xprep.correct_lon)
          .pipe(xprep.correct_coordinates)
          .pipe(xprep.parse_lon_lat_bounds)
          .pipe(xprep.maybe_convert_bounds_to_vertex)
          .pipe(xprep.maybe_convert_vertex_to_bounds)
          )
    if 'bnds' in ds.dims:
        ds = ds.drop_dims("bnds")
    if "vertex" in ds.dims:
        ds = ds.drop_dims("vertex")
    if ('lat' in ds.dims) | ('lon' in ds.dims):
        ds = xprep.combined_preprocessing(ds)
    if ds.x.max() > 180:
        ds = ds.assign_coords(
            {'x': (((ds['x'] + 180) % 360) - 180)}
        ).sortby('x')
    return ds


def open_dataset(f):
    """
    To make exception for MRI datasets with time units in
    days since 1850-01-01:
    ds = (
        xr.open_dataset(f, decode_times=False, **xr_kwargs)
        .sortby("time")
        .sel({'time': 
            [(x < 300000) & (x > -300000)
            for x in ds.time.values]
        })
    )
    """
    xr_kwargs = dict(
        chunks={},
    )
    try:
        ds = xr.open_dataset(f, **xr_kwargs)
        if ds.time.dt.calendar == "360_day":
            return ds.convert_calendar(
                calendar="standard",
                align_on="year",
            )
    except OverflowError as exc:
        ds = xr.open_dataset(f, decode_times=False, **xr_kwargs)
        if ds.time.attrs['units'] == 'days since 1850-01-01':
            ds = ds.sortby("time")
            ds = ds.sel({'time': [(x < 300000) & (x > 0.1)
                        for x in ds.time.values]})
            ds['time'] = [
                dt.datetime(1850, 1, 1) + dt.timedelta(x)
                for x in ds.time.values
            ]
        else:
            raise Exception('Time units not recognized.') from exc
    return ds


def process_one_dataset(ds):
    if 'zostoga' in ds.data_vars:
        cmip = xprep.rename_cmip6(ds).pipe(geo.drop_non_dim_coords)
    else:
        cmip = ds.pipe(xmip_wrapper).pipe(geo.drop_non_dim_coords)
    return cmip


def open_cmip(
    fnames: Path | list[Path],
    subset: dict[str, slice] = None,
) -> xr.Dataset:
    """Open netCDF files for one CMIP6 dataset.

    Args:
        fnames (list[pathlib.Path]):
            list of pathnames to netCDFs

    Returns:
        xr.Dataset:
            the concatenated CMIP6 dataset.
    """
    if not isinstance(fnames, list):
        return open_dataset(fnames).pipe(process_one_dataset)
    fnames_available = [f for f in fnames if f.stat().st_size > 0]
    if len(fnames_available) > 0:
        if subset is None:
            if len(fnames_available) < 2:
                return open_dataset(
                    fnames_available[0]
                ).pipe(process_one_dataset)
            return xr.concat(
                [open_dataset(f)
                 .pipe(process_one_dataset)
                 for f in fnames
                 ], dim="time"
            ).sortby("time")
        if len(fnames_available) < 2:
            return open_dataset(
                fnames_available[0]
            ).pipe(process_one_dataset
                   ).sel(**subset)
        return xr.concat(
            [open_dataset(f)
             .pipe(process_one_dataset)
             .sel(**subset)
             for f in fnames
             ], dim="time"
        ).sortby("time")
