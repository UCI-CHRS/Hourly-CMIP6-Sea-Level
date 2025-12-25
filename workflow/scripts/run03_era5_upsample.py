
"""Upsample ERA5 data"""

from pathlib import Path
import xarray as xr
from snakemake.script import snakemake
from aux import geo


def upsample_era5(
    f: Path,
    ref_cmip_grid_ds: xr.Dataset
) -> xr.Dataset:
    """Upsample one ERA5 netCDF file to a CMIP6 grid.

    Args:
        f (pathlib.Path):
            ERA5 netCDF file to upsample.
        f (pathlib.Path):
            path at which to save the upsampled ERA5 data.
        ref_cmip_grid_ds (xr.Dataset):
            CMIP6 grid to upsample to.

    Returns:
        xr.Dataset: upsampled dataset
    """
    ds_src = geo.open_ds(f, xvar="x")
    ds_match = ref_cmip_grid_ds.copy()
    regridded = geo.regrid_to_match(ds_src, ds_match, "x", "y", "x", "y")
    return regridded


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    processed_nc = Path(snakemake.input['processed_atmos'])
    ref_grid = Path(snakemake.input['ref_grid'])
    gridname = snakemake.params['gridname']
    year = snakemake.params['year']
    fname_out = snakemake.output[0]
    #-Script------------------------------------------------------------
    ds = upsample_era5(
        processed_nc,
        xr.open_dataset(ref_grid)
    )
    ds.to_netcdf(fname_out)

if __name__ == "__main__":
    main()
