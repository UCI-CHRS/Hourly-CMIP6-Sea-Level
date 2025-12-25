"""Check for missing CMIP files. """

from pathlib import Path
import warnings
import os
import yaml
import xarray as xr
import pandas as pd
from snakemake.script import snakemake


def parse_cmip_log(logfile: Path) -> dict[str, str]:
    """Read the log files generated during cmip6raw write.

    Args:
        logfile (Path): file containing logs for one master ID

    Returns:
        dict[str, str]: dictionary of missing search and file info
    """
    with open(logfile, "r") as f:
        log = yaml.safe_load(f)
    missing_mid = True
    files = [Path(x) for x in log.keys() if x[-3:] == ".nc"]
    if len(files) > 0:
        missing_mid = False
    return missing_mid, files, log


def check_corrupt(fname: Path) -> None:
    """Check if the file is readable. Remove if not.

    Args:
        fname (Path): file name to read

    Returns:
        None:
            Removes file if corrupt.
    """
    try:
        xr.open_dataset(
            fname, engine="netcdf4", decode_cf=False, decode_times=False,
            decode_coords=False, use_cftime=False, decode_timedelta=False
        )
    except (OSError, ValueError):
        os.remove(fname)


def glob_cmip(cmip_nc_dir: Path, mid: str) -> list[Path]:
    """Glob all CMIP files for one mid"""
    model, exp, ripf, res, var, grd = mid.split(".")[slice(3, 9)]
    files = list(cmip_nc_dir.glob(
        f"{var}_{res}_{model}_{exp}_{ripf}_{grd}*.nc"))
    return files


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    mid = snakemake.params['mid']
    logfile = snakemake.input['logfile']
    available = snakemake.output['available']
    missing = snakemake.output['missing']
    cmip_nc_dir = snakemake.params['cmip_nc_dir']
    #-Script------------------------------------------------------------
    # Read logs
    missing_mid, files, log = parse_cmip_log(logfile)
    # Check for files not captured in the log
    files += glob_cmip(cmip_nc_dir, mid)
    files = list({f.name for f in files})
    if len(files) > 0:
        missing_mid = False
    # Remove corrupt files
    # for file in files:
        # if cmip_nc_dir.joinpath(file).exists():
        #     check_corrupt(fname)
        # Make a table of missing MIDs and filenames
    if missing_mid:
        df_missing = pd.DataFrame({
            'fname': None,
            'mid': mid,
            'mid_missing': missing_mid
        }, index=[0])
    else:
        fnames_missing = [
            f for f in files if not cmip_nc_dir.joinpath(f).exists()]
        df_missing = pd.DataFrame({
            'mid': mid,
            'fname': fnames_missing,
            'mid_missing': missing_mid
        })
    df_missing.to_csv(missing)
    # Write out available files
    fnames_available = pd.DataFrame({
        'cmipfiles':
        [f for f in files if cmip_nc_dir.joinpath(f).exists()]
    })
    fnames_available.to_csv(available)


if __name__ == "__main__":
    main()
