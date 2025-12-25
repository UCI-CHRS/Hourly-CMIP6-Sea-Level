"""Set of utilities for reading and modifying geospatial data."""

import pathlib
import warnings

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import shapely
import xarray as xr
import xesmf as xe  # also requires esmpy
import xmip.preprocessing as xprep


dask.config.set(**{"array.slicing.split_large_chunks": True})


def mask_buffer(buffer: float, mymask: regionmask.Regions) -> regionmask.Regions:
    """Add a buffer to a regionmask object.

    Args:
        buffer (float):
            size of buffer to apply
        mymask (regionmask.Regions):
            mask object to buffer

    Returns:
        regionmask.Regions:
            mask with the buffer applied.
    """
    gdf = mymask.to_geodataframe()["geometry"].buffer(buffer, resolution=16)
    mymask_buffered = regionmask.Regions(outlines=gdf)
    return mymask_buffered


def inverse_mask(
    ds: xr.Dataset, mymask: regionmask.Regions, mask_val: int = 0
) -> xr.Dataset:
    """Returns the area outside of a mask.

    Args:
        ds (xr.Dataset):
            dataset to mask
        mymask (regionmask.Regions):
            mask object
        mask_val (int):
            Value of the inside of the mask

    Returns:
        xr.Dataset:
            dataset values only outside of the mask region.
    """
    mask = mymask.mask(ds, wrap_lon=False, use_cf=False)
    ds_mask = ds.where(mask != mask_val)
    return ds_mask


def open_ds(nc_fname: pathlib.Path, xvar: str) -> xr.Dataset:
    """Open a netCDF file and wrap longitudes -180:180 (instead of 0 to
    360) if needed.

    Args:
        nc_fname(pathlib.Path):
            path to the netCDF file
        xvar(str):
            name of the x-coordinate/longitude (probably "x" or "lon")

    Returns:
        xr.Dataset:
            xarray dataset with wrapped longitudinal coords
    """
    ds = xr.open_dataset(nc_fname)
    # Correct 0:360 lon to -180:180 if needed.
    ds = ds.assign_coords(
        {xvar: (((ds[xvar] + 180) % 360) - 180)}).sortby(xvar)
    return ds


def drop_non_dim_coords(ds):
    non_dim_coords = [x for x in list(ds.coords) if x not in list(ds.dims)]
    return ds.drop(non_dim_coords)


def gridded_dx_dy(
    ds: xr.Dataset, xvar: str = "x", yvar: str = "y"
) -> tuple[float, float]:
    """Get average dx and dy for a gridded xarray dataset ds.
    Round to 2 decimal points to avoid spurious unique values.

    Args:
        ds (xr.Dataset):
            gridded xarray dataset
        xvar (str):
            name of the x-coordinate
        yvar (str):
            name of the y-coordinate

    Returns:
        tuple[float, float]:
            (Median) dx and dy values
    """
    dx = np.unique(np.abs(np.diff(ds[xvar])))
    dy = np.unique(np.abs(np.diff(ds[yvar])))
    if (len(dx) > 1) or (len(dy) > 1):
        print("Warning: multiple dx or dy found in dataset. Using median.")
        return np.array([np.median(dx)]), np.array([np.median(dy)])
    else:
        if isinstance(dx, list):
            dx = dx[0]
        if isinstance(dy, list):
            dy = dy[0]
        return dx, dy


def ocean_mask(ds: xr.Dataset, **gridded_dx_dy_kwargs) -> xr.Dataset:
    """Create ocean mask for xarray dataset ds.
    Adds a buffer equal to distance between center of grid and corner of grid
    to account for grid cells within the bounds (with centers not in bounds).
    Note that the buffer won't work as intended if dy >> dx or dx >> dy.

    Args:
        ds (xr.Dataset):
            dataset to mask
        **gridded_dx_dy_kwArgs:
            any keyword arguments accepted by gridded_dx_dy()

    Returns:
        xr.Dataset:
            subset of ds that overlaps with the ocean (not continental land)
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    dx, dy = gridded_dx_dy(ds, **gridded_dx_dy_kwargs)
    if any(((len(dx) < 1), (len(dy) < 1))):  # only one grid cell
        print("Warning: only one grid cell. ocean_mask returning original dataset.")
        return ds
    else:
        buffer = 0.5 * np.sqrt(dx**2 + dy**2)
        mask_with_buffer = mask_buffer(-buffer, land)
        ds_mod = ds.rename({"x": "lon", "y": "lat"})
        ocean = inverse_mask(ds_mod, mask_with_buffer)
        ocean_mod = ocean.rename({"lon": "x", "lat": "y"})
        return ocean_mod


def match_extents(
    ds1_: xr.Dataset, ds2_: xr.Dataset, x1: str, y1: str, x2: str, y2: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """return overlapping areas of ds1 and ds2.
    NOTE: Consider replacing with xarray's reindex_like

    Args:
        ds1_ (xr.Dataset):
            one xarray dataset to overlap
        ds2_ (xr.Dataset):
            another xarray dataset to overlap with ds1_
        x1 (str):
            name of the x-coordinate in ds1_
        y1 (str):
            name of the y-coordinate in ds1_
        x2 (str):
            name of the x-coordinate in ds2_
        y2 (str):
            name of the y-coordinate in ds2_

    Returns:
        tuple[xr.Dataset, xr.Dataset]:
            subsets of ds1_ and ds2_ that overlap
    """
    ds1 = ds1_.rename({x1: "lon", y1: "lat"}).sortby("lat").sortby("lon")
    ds2 = ds2_.rename({x2: "lon", y2: "lat"}).sortby("lat").sortby("lon")
    # first clip src and match to smallest overlap
    lonmin = np.max([ds1.lon.values.min(), ds2.lon.values.min()])
    lonmax = np.min([ds1.lon.values.max(), ds2.lon.values.max()])
    latmin = np.max([ds1.lat.values.min(), ds2.lat.values.min()])
    latmax = np.min([ds1.lat.values.max(), ds2.lat.values.max()])
    ds1 = ds1.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
    ds2 = ds2.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
    return ds1, ds2


def nc_box(
    ds: xr.Dataset, xvar: str = "x", yvar: str = "y"
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Get dx and dy for a gridded xarray dataset ds when bounds are
    not available.

    Args:
        ds (xr.Dataset):
            dataset for which to get dx and dy
        xvar (str):
            name of the x-coordinate in ds
        yvar (str):
            name of the y-coordinate in ds

    Returns:
        tuple[xr.Dataset, pd.DataFrame]:
            dataset and dataframe versions of the lower and upper x and y
            bounds of each grid coordinate (xl, xu, yl, yu)
    """
    x, y = ds[xvar], ds[yvar]

    def fun(p):
        pl, pu = np.full(len(p), np.nan), np.full(len(p), np.nan)
        d = np.abs(np.diff(p))
        pl[1:] = p[1:] - d / 2
        pu[:-1] = p[:-1] + d / 2
        # Assume edge points are equally centered
        pl[0] = p[0] - d[0] / 2
        pu[-1] = p[-1] + d[-1] / 2
        return pl, pu

    xl, xu = fun(x.values)
    yl, yu = fun(y.values)
    # Add upper and lower bounds to indexed xarray dataset

    def dax(lv, lvn): 
        return xr.Dataset(
            data_vars={lvn: xr.DataArray(lv, coords={xvar: ds[xvar].values})}
        )

    def day(lv, lvn): 
        return xr.Dataset(
            data_vars={lvn: xr.DataArray(lv, coords={yvar: ds[yvar].values})}
        )
    xll, yll = xr.broadcast(dax(xl, "xl"), day(yl, "yl"))
    xuu, yuu = xr.broadcast(dax(xu, "xu"), day(yu, "yu"))
    dss = xr.combine_by_coords([xll, yll, xuu, yuu])
    df_grid = dss.to_dataframe().reset_index()
    return dss, df_grid


def netcdf_cell_geometry(
    ds: xr.Dataset, x: str, y: str, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Get .nc grid cell geometry as GeoDataFrame boxes.
        Plot to check that the grid and cell centers are aligned:
        grid_cell_centers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df_grid[x], df_grid[y]))
        ax = cell.plot(facecolor="none", edgecolor='grey')
        grid_cell_centers.plot(ax=ax)

    Args:
        ds (xr.Dataset):
            xarray datset for which to get the cell geometry
        x (str):
            name of x spatial dimension
        y (str):
            name of y spatial dimension
        crs (str):
            CRS authority string readable by the geopandas crs keyword argument

    Returns:
        gpd.GeoDataFrame:
            geopandas cell geometry corresponding to the ds grid.
    """
    _, df_grid = nc_box(ds, x, y)
    grid_cells = [
        (x, y, shapely.geometry.box(xl, yl, xu, yu))
        for x, y, xl, yl, xu, yu in zip(
            df_grid["x"],
            df_grid["y"],
            df_grid["xl"],
            df_grid["yl"],
            df_grid["xu"],
            df_grid["yu"],
        )
    ]
    cell = gpd.GeoDataFrame(grid_cells, columns=[
                            "x", "y", "geometry"], crs=crs)
    return cell


def point_to_grid(
    match_ds: xr.Dataset,
    point_df: pd.DataFrame,
    point_df_vars: list[str],
    csv_time_var: str = "time",
    point_df_xvar: str = "lon",
    point_df_yvar: str = "lat",
    crs: str = "EPSG:4326",
    **netcdf_cell_geometry_kwargs
) -> gpd.GeoDataFrame:
    """Regrid point data to a coarse grid with GDAL using moving average
        algorithm.
        https://gdal.org/programs/gdal_grid.html#average
        See sample code in notebooks/gdal_grid.ipynb.
        NOTE: Consider replacing with xarray's interp_like

    Args:
        match_ds (xr.Dataset):
            dataset with the target grid
        point_df (pd.DataFrame):
            dataframe with lat and lon data
        point_df_var (str):
            non-geometric variable in point_df to keep
        csv_time_var (str):
            time variable in point_df
        **netcdf_cell_geometry_kwArgs:
            any keyword arguments accepted by netcdf_cell_geometry()

    Returns:
        gpd.GeoDataFrame:
            georeferenced version of point_df mapped to the coarse grid of match_ds
    """
    gdf = gpd.GeoDataFrame(
        point_df,
        geometry=gpd.points_from_xy(
            point_df[point_df_xvar], point_df[point_df_yvar]
        ),  # point_df.lon, point_df.lat),
        crs=crs,
    )
    gdf_grouped = gdf.groupby(csv_time_var)
    # Get .nc grid cell geometry
    cell = netcdf_cell_geometry(match_ds, crs=crs, **netcdf_cell_geometry_kwargs)
    def one_time_step(sliced_gdf):
        # Join dataframes
        gdf_ = sliced_gdf[["geometry"] + point_df_vars]
        cell_ = cell.copy()
        merged = gpd.sjoin(gdf_, cell_, how="left", predicate="within")
        # Aggregate station data within each cell.
        dissolve = merged.dissolve(by="index_right", aggfunc="mean")
        cell_.loc[dissolve.index, point_df_vars] = dissolve[point_df_vars].values
        # Remove empty cells
        return cell_.dropna()
    # Apply to all time steps
    gridded = gdf_grouped.apply(
        one_time_step).reset_index().drop("level_1", axis=1)
    return gridded


# gdf_cell_geom: gpd.GeoDataFrame) -> xr.Dataset:
def cell_geometry_to_xarray(gdf):
    """Converts geodataframe with box geometry into xarray dataset.
    Requires columns "x" and "y"; otherwise these will be computed from the
    box geometry centroid, which could be inaccurate.

    Args:
        gdf_cell_geom (gpd.GeoDataFrame):
            geopandas geometry to convert to xarray

    Returns:
        xr.Dataset:
            xarray dataset with the same geometry as gdf_cell_geom
    """
    # gdf = gdf_cell_geom.copy()
    # if not all([dim  in gdf.columns for dim in ['x', 'y']]):
    #     centroids = gdf.centroid
    #     gdf['x'] = centroids.geometry.x
    #     gdf['y'] = centroids.geometry.y
    # gdf = gdf.drop("geometry", axis = 1).sort_index()
    xds = (
        gdf.drop("geometry", axis=1)
        .sort_index()
        .set_index(["time", "x", "y"])
        .to_xarray()
    )
    return xds


def regrid_to_match(
    ds_src: xr.Dataset,
    ds_match: xr.Dataset,
    src_x: str,
    src_y: str,
    match_x: str,
    match_y: str,
) -> xr.Dataset:
    """Regrids source .nc file to match CMIP .nc file resolution.
        Uses XMIP to preprocess the match CMIP file.
        Uses the xesmf module. "conservative" is a simple area-weighted average.
        Conservative and conservative_normed methods require grid corners, which
        are automatically estimated by the cf_xarray add_bounds method; see
        https://github.com/pangeo-data/xESMF/blob/530b804c28a9b4f64dd360384897c4c2b34ab8c3/xesmf/frontend.py#L85
        NOTE: Consider replacing with xarray's interp_like

    Args:
        ds_src (xr.Dataset):
            source dataset to regrid
        ds_match (xr.Dataset):
            dataset with the target grid/resolution
        src_x (str):
            x-coordinate name of ds_src
        src_y (str):
            y-coordinate name of ds_src
        match_x (str):
            x-coordinate name of ds_match
        match_y (str):
            y-coordinate name of ds_match

    Returns:
        xr.Dataset:
            ds_src data regridded onto the ds_match grid.
    """
    # first clip src and match to smallest overlap
    ds_src_, ds_match_ = match_extents(
        ds_src, ds_match, src_x, src_y, match_x, match_y)
    ds_src_ = ds_src_.drop_duplicates(dim=["lat", "lon"])
    ds_match_ = ds_match_.drop_duplicates(dim=["lat", "lon"])
    # create the regridder
    source = xr.Dataset({"lat": ds_src_["lat"], "lon": ds_src_["lon"]}).rename(
        {'lat': "latitude", "lon": "longitude"})
    match = xr.Dataset({"lat": ds_match_["lat"], "lon": ds_match_["lon"]}).rename(
        {'lat': "latitude", "lon": "longitude"})
    regridder = xe.Regridder(source, match, "conservative_normed")
    ds_src_match = regridder(ds_src_, skipna=True)
    ds_src_match = ds_src_match.rename({"longitude": src_x, "latitude": src_y})
    return ds_src_match
