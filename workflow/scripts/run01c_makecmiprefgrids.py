
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from snakemake.script import snakemake
from aux import cmip, mwm


def get_reference_cmip_grid(
    cmip_nc_dir: Path,
    metadata_df: pd.DataFrame,
    grid_id: str
) -> xr.DataArray | None:
    """Returns a data array with the x, y coordinates of one dataset that
        has the grid grid_id.
        NOTE: grid_id is the shortened grid id produced by grid_metadata.

    Args:
        grid_id (str):
            one of the shortened grid ids

    Returns:
        xr.DataArray:
            xarray object with the x, y coordinates of grid_id
    """
    md = metadata_df.loc[metadata_df['grid_id'] == grid_id, :]
    ind = 0
    fpath = []
    while (len(fpath) < 1) and (ind < md.shape[0] - 1):
        ind += 1
        row = md.iloc[ind, :]
        var = row['variable_id']
        model = row['source_id']
        ripf = row['variant_label']
        fpath = [
            x for x in cmip_nc_dir.glob(f"{var}_*{model}*_{ripf}*.nc")
            if x.exists()]
    if len(fpath) < 1:
        print(f"Warning: no files found for grid {grid_id}")
        return None
    else:
        sample_path = [
            f for f in fpath
            if f.stat().st_size == min([f.stat().st_size for f in fpath])
        ][0]
        ds = cmip.open_cmip(sample_path)[var]
        da = ds.isel(time=0).drop_vars("time")
        da.values.fill(np.nan)
        return da


def make_reference_grid(
    cmip_nc_dir: Path,
    metadata_df: pd.DataFrame,
    grid_id: str,
    grid_dir: Path
) -> xr.DataArray | None:
    """Writes the dataarray from _get_reference_cmip_grid(grid_id) to disk.
    NOTE: grid_id is the shortened grid id produced by grid_metadata.

    Args:
        grid_id (str): one of the grid_ids

    Returns:
        xr.DataArray: xarray object with the x, y coordinates of grid_id
    """
    grid_fname = grid_dir.joinpath(f"{grid_id}.nc")
    ds = get_reference_cmip_grid(cmip_nc_dir, metadata_df, grid_id)
    if ds is not None:
        ds.to_netcdf(grid_fname)
    return ds


def subset_cmip_grids_to_tide_gauges(x, y, cmip_x, cmip_y, grid, n_grids):
    df = pd.DataFrame({'x': x, 'y': y}).drop_duplicates()
    # Moving window
    coords = [{'x': lon, 'y': lat} for lon in cmip_x for lat in cmip_y
              if (min(mwm.make_spatial_moving_window(
                  df, lat, lon, grid, n_grids, apply_ocean_mask=False
              ).shape) > 0)]
    return coords


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    grid_dir = snakemake.params['upsampled_grids_dir']
    cmip_nc_dir = snakemake.params['rawdata_cmip_nc_dir']
    cmip_metadata_f = snakemake.input['cmip_mids']
    tide_gauges_f = snakemake.input['tide_gauges']
    n_grids = snakemake.params['ngrids']
    grid_id = snakemake.params['grid_id']
    #-Script------------------------------------------------------------
    # Write CMIP ref grids
    metadata_df = pd.read_csv(cmip_metadata_f)
    ref_grid = make_reference_grid(cmip_nc_dir, metadata_df, grid_id, grid_dir)
    # Write each grid to .csv with only lat/lons corresponding to the tide gauges
    stns = pd.read_csv(tide_gauges_f)
    x, y = stns['lng'], stns['lat']
    if ('x' in ref_grid.coords) and ('y' in ref_grid.coords):
        cmip_x = ref_grid.x.values
        cmip_y = ref_grid.y.values
        if isinstance(cmip_x[0], float) and isinstance(cmip_y[0], float):
            csv_fname = grid_dir.joinpath(
                f"{grid_id}_ngrids_{n_grids}_noaastns.csv")
            coords = subset_cmip_grids_to_tide_gauges(
                x, y, cmip_x, cmip_y, ref_grid, n_grids)
            pd.DataFrame(coords).to_csv(csv_fname)


if __name__ == "__main__":
    main()
