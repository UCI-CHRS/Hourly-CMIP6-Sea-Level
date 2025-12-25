"""Regrid CMIP to largest nominal standard grid """

from pathlib import Path
import xarray as xr
from snakemake.script import snakemake
from aux import geo


def regrid_cmip(ds_cmip: xr.Dataset, ds_refgrid: xr.Dataset):
    return geo.regrid_to_match(
        ds_src=ds_cmip,
        ds_match=ds_refgrid,
        src_x="x",
        src_y="y",
        match_x="x",
        match_y="y",
    )


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    zos_in = xr.open_dataset(snakemake.input['zos'])
    zostoga_in = xr.open_dataset(snakemake.input['zostoga'])
    uas_in = xr.open_dataset(snakemake.input['uas'])
    vas_in = xr.open_dataset(snakemake.input['vas'])
    psl_in = xr.open_dataset(snakemake.input['psl'])
    ref_grid = xr.open_dataset(snakemake.input['ref_grid'])
    daily_out = snakemake.output['daily']
    monthly_out = snakemake.output['monthly']
    #-Script------------------------------------------------------------
    # Upsample individual
    zos_upsampled = regrid_cmip(zos_in, ref_grid)
    uas_upsampled = regrid_cmip(uas_in, ref_grid)
    vas_upsampled = regrid_cmip(vas_in, ref_grid)
    psl_upsampled = regrid_cmip(psl_in, ref_grid)
    # Merge by frequency
    monthly_upsampled = xr.merge(
        [zos_upsampled['zos'],
         zostoga_in['zostoga'].squeeze().broadcast_like(zos_upsampled['zos'])]
    )
    daily_upsampled = xr.merge(
        [uas_upsampled,
         vas_upsampled,
         psl_upsampled
        ]
    )
    # Write to files
    daily_upsampled.to_netcdf(daily_out)
    monthly_upsampled.to_netcdf(monthly_out)


if __name__ == "__main__":
    main()
