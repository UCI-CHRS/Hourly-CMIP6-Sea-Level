
""" Moving window mapping functions. """

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from collections.abc import Callable
from . import geo

# Define spatial, temporal, and spatiotemporal windows ---------------


def window_size(ref_grid: xr.Dataset, n_grids: int, **gridded_dx_dy_kwargs) -> float:
    """Calculates square window size based on a multiple of dataset ds size. 

    Args: 
        ref_grid (xr.Dataset): 
            dataset with coordinate lat x lon x time that the window is based on
        n_grids (int): 
            number of CMIP grids along one dimension of the window
        **gridded_dx_dy_kwArgs:
             any keyword arguments accepted by src.aux.geo.gridded_dx_dy()

    Returns: 
        float: 
            window size along one dimension in the spatial units of ds
    """
    dx, dy = geo.gridded_dx_dy(ref_grid, **gridded_dx_dy_kwargs)
    window = float(n_grids) * np.max([dx, dy])
    return window


def spatial_moving_window(lat: float, lon: float, ref_grid: xr.Dataset,
                          n_grids: int) -> dict[str, float]:
    """Define the 3D bounding box (x x y x time) of the spatiotemoral moving window. 

    Args: 
        lat (float):
            central latitude in the window
        lon (float):
            central longitude in the window
        ref_grid (xr.Dataset): 
            dataset with the CMIP grid on which the window size is based
        n_grids (int): 
            number of CMIP grids along one dimension of the window

    Returns: 
        dict[str, float]: 
            dictionary of xmin, xmax, ymin, ymax of the moving window box
    """
    windowsize = window_size(ref_grid, n_grids)
    bbox = {
        'xmin': lon - windowsize/2,
        'xmax': lon + windowsize/2,
        'ymin': lat - windowsize/2,
        'ymax': lat + windowsize/2,
    }
    return bbox


def spatiotemporal_moving_window(lat: float, lon: float, month: int, day: int,
                                 hour: int, ref_grid: xr.Dataset, n_grids: int = 5,
                                 n_days: int = 30) -> dict:
    """Define the 3D bounding box (x x y x time) of the spatiotemoral moving window. 

    Args: 
        lat (float):
            central latitude in the window
        lon (float):
            central longitude in the window
        month (int):
            month of the central timestamp
        day (int): 
            day of the central timestamp
        hour (int): 
            hour of the central timestamp
        ref_grid (xr.Dataset): 
            dataset with the CMIP grid on which the window size is based
        n_grids (int): 
            number of CMIP grids along one dimension of the window
        n_days (int): 
            number of days in the moving window

    Returns: 
        dict: 
            dictionary of xmin, xmax, ymin, ymax, tmin, tmax of the moving window box
    """
    windowsize = window_size(ref_grid, n_grids)
    def myt(yr): return dt.datetime(year=yr, month=month, day=day, hour=hour)

    def bbox(yr): return {
        'xmin': lon - windowsize/2,
        'xmax': lon + windowsize/2,
        'ymin': lat - windowsize/2,
        'ymax': lat + windowsize/2,
        'tmin': myt(yr) - dt.timedelta(days=(np.ceil(n_days/2))),
        'tmax': myt(yr) + dt.timedelta(days=(np.ceil(n_days/2))),
    }
    return bbox


def check_central_location(ds: xr.Dataset, lat: float, lon: float,
                           bbox: dict[str, float],
                           xvar: str = "x", yvar: str = "y"
                           ) -> bool:
    """Verify that the central location is within the range of the dataset. 

    Args: 
        ds (xr.Dataset): 
            gridded dataset with dimensions x, y, time
        lat (float):
            latitude of the central location coordinates
        lon (float):
            longitude of the central location coordinates
        bbox (dict[str, float]):
            dictionary of xmin, xmax, ymin, ymax 
        xvar (int):
            name of the x-coordinate in ds_geo
        yvar (int): 
            name of the y-coordinate in ds_geo

    Returns: 
        bool: if True, the central location falls within the bounds of ds_geo
    """
    xs = ds[xvar]
    ys = ds[yvar]
    is_in = (lon >= min(xs)) & (lon <= max(xs)) & (
        lat >= min(ys)) & (lat <= max(ys))
    # Check that it has ocean values
    if is_in:
        ds_clip = (ds.sortby('x').sortby('y')
                   .sel(x=slice(bbox['xmin'], bbox['xmax']),
                        y=slice(bbox['ymin'], bbox['ymax'])))
        ds_geo_ocean = geo.ocean_mask(ds_clip)
        return all(np.array(list(ds_geo_ocean.sizes.values())) > 0)
    else:
        return False

# Make spatial moving window -----------------------------------------


def make_spatial_moving_window_xr(ds: xr.Dataset,
                                  bbox: dict,
                                  apply_ocean_mask: bool = True) -> xr.Dataset:
    """Defines moving window size and locations.
        central_location: dictionary of "lat" and "lon" coordinates of central point
        window_size_km: width and height of moving window. 
        This squared region is assumed to be statistically homogeneous. 
        Window size should be in the geospatial units of the dataset 
        (e.g., degrees lat/lon or km.)
        NOTE: ds spatial units should be "lat" and "lon", with lon values
        bounded between -180:180 (i.e., CMIP datasets should already 
        be preprocessed). 

    Args: 
        ds (xr.Dataset): 
            unclipped dataset
        bbox (dict): 
            xmin, xmax, ymin, ymax bounding box of the spatial moving 
            window (e.g., output of spatial_moving_window)
        apply_ocean_mask (bool): 
            Whether to mask out land data. Default is true (set to false for tide gauge data.)

    Returns: 
        xr.Dataset:
            ds clipped to the spatial moving window (with mask applied)
    """
    # ds_clip = (ds.sortby('x').sel(x = slice(bbox['xmin'], bbox['xmax']))
    #              .sortby('y').sel(y = slice(bbox['ymin'], bbox['ymax'])))
    ds_clip = ds.sel(x=slice(bbox['xmin'], bbox['xmax']), y=slice(
        bbox['ymin'], bbox['ymax']))
    # clip subsample bbox with additional mask
    if apply_ocean_mask:
        ds_clip_masked = geo.ocean_mask(ds_clip)
        return ds_clip_masked
    else:
        return ds_clip


def make_spatial_moving_window_df(df: pd.DataFrame,
                                  bbox: dict[str, float]) -> pd.DataFrame:
    """Slice dataset centered at one time step and gauge coordinate. 

    Args: 
        df (pd.DataFrame with "x" and "y" columns): 
            dataframe to clip
        bbox (dict): 
            xmin, xmax, ymin, ymax bounding box of the spatial moving 
            window (e.g., output of spatial_moving_window)

    Returns: 
        pd.DataFrame:
            df clipped to the spatial window.
    """
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    xmax = bbox['xmax']
    ymax = bbox['ymax']
    df_clip = df.loc[(df['x'] >= xmin) & (df['x'] <= xmax) &
                     (df['y'] >= ymin) & (df['y'] <= ymax), :]
    return df_clip


def make_spatial_moving_window(ds: xr.Dataset | pd.DataFrame, lat, lon,
                               ref_grid, n_grids, **xr_kwargs):
    bbox = spatial_moving_window(lat, lon, ref_grid, n_grids)
    if isinstance(ds, xr.Dataset) | isinstance(ds, xr.DataArray):
        # kwargs = optional apply_ocean_mask argument
        return make_spatial_moving_window_xr(ds, bbox, **xr_kwargs)
    elif isinstance(ds, pd.DataFrame) | isinstance(ds, pd.Series):
        return make_spatial_moving_window_df(ds, bbox)

# Make temporal moving window ----------------------------------------

def make_temporal_moving_window_xr(
    ds: xr.Dataset,
    years: range,
    month: int,
    day: int,
    hour: int,
    n_days: int
) -> xr.Dataset:
    """Define a moving temporal window centered around each datetime 
    within a year corresponding to (month, day, hour). 
    NOTE: Time dimension should be called 'time'. 

    Args: 
        ds (xr.Dataset): 
            dataset to clip
        month (int): 
            month of central timestamp
        day (int):  
            day of central timestamp
        hour (int): 
            hour of central timestamp
        n_days (int): 
            number of days in the moving window
        years (range): 
            range of start to end year

    Returns:
        xr.Dataset:
            ds clipped to the temporal moving window. 
    """
    # Ensure datetime64[ns] type
    ds = ds.sortby("time")
    time = pd.to_datetime(ds.time.values)
    # Compute valid window centers
    centers = []
    for year in years:
        try:
            centers.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
        except ValueError:
            # Handle Feb 29 on non-leap years
            if month == 2 and day == 29:
                centers.append(pd.Timestamp(year=year, month=3, day=1, hour=hour))
            else:
                continue
    centers = np.array(centers, dtype='datetime64[ns]')
    half_window = np.ceil(n_days / 2).astype(int)
    deltas = np.arange(-half_window, half_window + 1, dtype='timedelta64[D]')
    # Create all days in all windows (flattened)
    all_times = (centers[:, None] + deltas[None, :]).ravel()
    # Clip to dataset time bounds
    all_times = all_times[(all_times >= time.min()) & (all_times <= time.max())]
    # Use numpy intersection for fast filtering
    valid_mask = np.isin(time.values, all_times)
    return ds.isel(time=valid_mask)


def make_temporal_moving_window_df(
    df: pd.DataFrame,
    years: range,
    month: int,
    day: int,
    hour: int,
    n_days: int
) -> pd.DataFrame:
    """Slice dataset centered at one time step
    NOTE: Uses March 1 values for Feb 29 on non-leap years. 

    Args: 
        df (pd.DataFrame):
            dataframe to clip
        years (range): 
            range of start to end year
        month (int): 
            month of central timestamp
        day (int):  
            day of central timestamp
        hour (int): 
            hour of central timestamp
        n_days (int): 
            number of days in the moving window

    Returns: 
        pd.DataFrame:
            df clipped to the spatiotemporal moving window
    """
     # --- ensure datetime index ---
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if df.index.empty:
        return df.copy()
    df = df.sort_index()
    tmin_df, tmax_df = df.index.min(), df.index.max()
    # --- build valid centers ---
    centers = []
    for year in years:
        try:
            centers.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
        except ValueError:
            if month == 2 and day == 29:
                centers.append(pd.Timestamp(year=year, month=3, day=1, hour=hour))
            else:
                continue
    if not centers:
        return df.iloc[0:0].copy()
    # --- construct window bounds ---
    half = pd.Timedelta(days=np.ceil(n_days / 2))
    bounds = np.array([(c - half, c + half) for c in centers], dtype="datetime64[ns]")
    # Clip bounds to dataset range
    bounds[:, 0] = np.maximum(bounds[:, 0], tmin_df.to_datetime64())
    bounds[:, 1] = np.minimum(bounds[:, 1], tmax_df.to_datetime64())
    # --- merge overlapping or adjacent windows to minimize scans ---
    # Sort by start time
    bounds = bounds[np.argsort(bounds[:, 0])]
    merged = []
    cur_start, cur_end = bounds[0]
    for start, end in bounds[1:]:
        if start <= cur_end:  # overlapping
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    # --- create final mask ---
    times = df.index.values
    mask = np.zeros(len(times), dtype=bool)
    for start, end in merged:
        mask |= (times >= start) & (times <= end)
    result = df.loc[mask].copy()
    return result


def make_temporal_moving_window(ds: xr.Dataset, years: range,
                                month: int, day: int, hour: int,
                                n_days: int) -> xr.Dataset:
    if isinstance(ds, xr.Dataset) | isinstance(ds, xr.DataArray):
        # kwargs = optional apply_ocean_mask argument
        return make_temporal_moving_window_xr(ds, years, month, day, hour, n_days)
    elif isinstance(ds, pd.DataFrame) | isinstance(ds, pd.Series):
        return make_temporal_moving_window_df(ds, years, month, day, hour, n_days)

# Make spatiotemporal moving window ----------------------------------


def make_spatiotemporal_moving_window_xr(df: xr.Dataset, lat: float, lon: float, years: range,
                                         month: int, day: int, hour: int,
                                         ref_grid: xr.Dataset, n_grids: int = 5,
                                         n_days: int = 30) -> pd.DataFrame:
    """Slice dataset centered at one time step and gauge coordinate. 
        NOTE: Uses March 1 values for Feb 29 on non-leap years. 

    Args: 
        df (pd.DataFrame):
            dataset to clip
        lat (float):
            latitude of central window coordinate
        lon (float):
            longitude of central window coordinate
        month (int): 
            month of central timestamp
        day (int):  
            day of central timestamp
        hour (int): 
            hour of central timestamp
        years (range): 
            range of start to end year
        ref_grid (xr.Dataset):
            gridded dataset on which the moving window size is based
        n_grids (int):
            number of ref_ds grids in the spatial moving window
        n_days (int): 
            number of days in the moving window

    Returns: 
        pd.DataFrame:
            df clipped to the spatiotemporal moving window
    """
    bbox = spatial_moving_window(lat, lon, ref_grid, n_grids)
    df_clip = make_spatial_moving_window_xr(df, bbox)
    df_clip_masked_time_slice = make_temporal_moving_window_xr(
        df_clip, years, month, day, hour, n_days
    )
    return df_clip_masked_time_slice


def make_spatiotemporal_moving_window_df(df: pd.DataFrame, lat: float, lon: float, years: range,
                                         month: int, day: int, hour: int,
                                         ref_grid: xr.Dataset, n_grids: int = 5,
                                         n_days: int = 30) -> pd.DataFrame:
    """Slice dataset centered at one time step and gauge coordinate. 
        NOTE: Uses March 1 values for Feb 29 on non-leap years. 

    Args: 
        df (pd.DataFrame):
            dataframe to clip
        lat (float):
            latitude of central window coordinate
        lon (float):
            longitude of central window coordinate
        month (int): 
            month of central timestamp
        day (int):  
            day of central timestamp
        hour (int): 
            hour of central timestamp
        years (range): 
            range of start to end year
        ref_grid (xr.Dataset):
            gridded dataset on which the moving window size is based
        n_grids (int):
            number of ref_ds grids in the spatial moving window
        n_days (int): 
            number of days in the moving window

    Returns: 
        pd.DataFrame:
            df clipped to the spatiotemporal moving window
    """
    df_clip = make_spatial_moving_window_df(df, lat, lon, ref_grid, n_grids)
    df_clip_masked_time_slice = make_temporal_moving_window_df(
        df_clip, years, month, day, hour, n_days
    )
    return df_clip_masked_time_slice


def make_spatiotemporal_moving_window(ds, lat: float, lon: float, years: range,
                                      month: int, day: int, hour: int,
                                      ref_grid: xr.Dataset, n_grids: int = 5,
                                      n_days: int = 30):
    if isinstance(ds, xr.Dataset) | isinstance(ds, xr.DataArray):
        # kwargs = optional apply_ocean_mask argument
        return make_temporal_moving_window_xr(ds, years, month, day, hour, n_days)
    elif isinstance(ds, pd.DataFrame) | isinstance(ds, pd.Series):
        return make_temporal_moving_window_df(ds, years, month, day, hour, n_days)
