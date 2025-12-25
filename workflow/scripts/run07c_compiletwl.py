
from pathlib import Path
import os
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from snakemake.script import snakemake


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs--------------------------------
    tides_file = snakemake.input['tides_hourly']
    msl_file = snakemake.input['msl_hourly']
    ntr_file = snakemake.input['ntr_hourly']
    nc_out = snakemake.output[0]
    # -Script-----------------------------------------------------------
    msl = xr.open_dataset(msl_file)
    tides = xr.open_dataset(tides_file)
    ntr = xr.open_dataset(ntr_file)
    twl = xr.Dataset({'twl': msl['msl'] + tides['tides'] + ntr['ntr']})
    twl.attrs = {
        'twl': "Total water level, sum of msl, tides, ntr."
    }
    twl.to_netcdf(nc_out)


if __name__ == "__main__":
    main()
