"""Regrid CMIP to largest nominal standard grid """

from pathlib import Path
import warnings
import pandas as pd
import xarray as xr
import numpy as np
from snakemake.script import snakemake
from aux import cmip


def optimize_cmip6_encoding(
    ds: xr.Dataset,
    target_chunk_mb: int = 30,
    chunk_overrides: dict | None = None,
    compression: bool = True,
    complevel: int = 4
) -> dict:
    """Return an encoding dict for writing CMIP6 subsets faster.

    Args:
    ds (xr.Dataset):
        Dataset to write (should already be re-chunked in memory with ds.chunk()).
    target_chunk_mb (int, default 30):
        Desired approximate chunk size in MB.
    chunk_overrides (dict, optional):
        Dict of {dim: chunk_size} to override automatic chunk sizes.
        Use -1 for "all elements along this dimension".
    compression (bool, default True):
        Whether to apply zlib compression to the output.
    complevel (int, default 4):
        Compression level (0-9). Lower = faster.

    Returns:
    dict:
        Encoding dictionary suitable for passing to to_netcdf().
    """
    # Required for saving with NetCDF4 engine
    valid_keys = {
        "_FillValue", "complevel", "zlib", "contiguous", "endian",
        "significant_digits", "least_significant_digit", "dtype",
        "chunksizes", "quantize_mode", "shuffle", "fletcher32",
        "szip_coding", "szip_pixels_per_block"
    }
    encoding = {}
    for var in ds.data_vars:
        var_enc = {}
        # Get element size in bytes
        dtype_size = np.dtype(ds[var].dtype).itemsize
        # Decide chunk sizes for each dim
        chunksizes = []
        for dim in ds[var].dims:
            if chunk_overrides and dim in chunk_overrides:
                size = chunk_overrides[dim]
                if size == -1:
                    size = ds.sizes[dim]
            else:
                if dim in ("x", "y"):
                    # Keep all spatial points in one chunk
                    size = ds.sizes[dim]
                elif dim == "time":
                    # Auto choose time chunk size
                    n_space = np.prod([ds.sizes[d]
                                      for d in ds[var].dims if d != "time"])
                    bytes_per_step = n_space * dtype_size
                    size = max(1, int(target_chunk_mb *
                               1024**2 // bytes_per_step))
                    size = min(size, ds.sizes[dim])  # can't exceed dim length
                else:
                    size = ds.sizes[dim]
            chunksizes.append(size)
        var_enc["chunksizes"] = tuple(chunksizes)
        # Compression settings
        if compression:
            var_enc.update({
                "zlib": True,
                "complevel": complevel,
                "shuffle": True
            })
        else:
            var_enc["zlib"] = False
        # Keep only valid keys
        var_enc = {k: v for k, v in var_enc.items() if k in valid_keys}
        encoding[var] = var_enc
    return encoding


def write_nc(
    ds: xr.Dataset,
    fname: str | Path
) -> None:
    """Speed up CMIP write times with optimize_cmip6_encoding 
    and larger chunk sizes. """
    # Build optimized encoding
    encoding = optimize_cmip6_encoding(ds)
    # Write fast
    ds = ds.compute()
    ds.to_netcdf(
        fname,
        engine="netcdf4",
        encoding=encoding
    )


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    available_files = (
        pd.read_csv(snakemake.input['available_files'])['fname'].tolist()
    )
    # out files
    fname_out = snakemake.output[0]
    # params
    model = snakemake.params['model']
    ripf = snakemake.params['ripf']
    exp = snakemake.params['exp']
    variable = snakemake.params['variable']
    cmip_nc_dir = snakemake.params['cmip_nc_dir']
    bbox_conus = snakemake.params['bbox_conus']
    bbox_enso = snakemake.params['bbox_enso']
    if variable == "tos":
        xmin, ymin, xmax, ymax = (
            bbox_enso[name] for name in ['xmin', 'ymin', 'xmax', 'ymax']
        )
    else:
        xmin, ymin, xmax, ymax = (
            bbox_conus[name] for name in ['xmin', 'ymin', 'xmax', 'ymax']
        )
    #-Script------------------------------------------------------------
    def getfiles(var):
        return [
            cmip_nc_dir.joinpath(f) for f in available_files
            if (
                all(x in f for x in (var, model, ripf, exp)) and
                int(f.split("_")[-1][:4]) < 2100
            )
        ]

    if variable == "zostoga":
        ds = (
            cmip.open_cmip(getfiles("zostoga"))
        )
    else:
        if variable == "zos":
            files = getfiles("zos_")
        else:
            files = getfiles(variable)
        ds = (
            cmip.open_cmip(
                files,
                subset=dict(
                    x=slice(xmin, xmax),
                    y=slice(ymin, ymax)
                ))
        )
    write_nc(ds, fname_out)


if __name__ == "__main__":
    main()
