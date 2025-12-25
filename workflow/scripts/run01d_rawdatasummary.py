
from pathlib import Path
import os
import pandas as pd
from snakemake.script import snakemake


def read_files(fnames_in: list[str]) -> tuple[list[pd.DataFrame]]:
    """Read the individual _missing.csv and _available.csv files. 

    Args:
        fnames_in (list[str]): list of filenames for all stations/years/etc

    Returns:
        tuple[list[pd.DataFrame]]:
            Lists of missing and available dataframes read from fnames_in
    """
    missing = [
        pd.read_csv(f)
        for f in fnames_in
        if ("_missing.csv" in f) and (os.path.getsize(f) > 0)
    ]
    available = [
        pd.read_csv(f)
        for f in fnames_in
        if ("_available.csv" in f) and (os.path.getsize(f) > 0)
    ]
    return missing, available


def parsecmip(fname):
    var, _, model, exp, ripf, _, _ = fname.split("_")
    return {'fname': fname, 'var': var, 'model': model, 'exp': exp, 'ripf': ripf}


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    cmip_in = snakemake.input['cmip']
    atmos_in = snakemake.input['atmos']
    ocean_in = snakemake.input['ocean']
    ocean_missing_out = snakemake.output['ocean_missing']
    cmip_available_out = snakemake.output['cmip_available']
    cmip_missing_out = snakemake.output['cmip_missing']
    atmos_available_out = snakemake.output['atmos_available']
    atmos_missing_out = snakemake.output['atmos_missing']
    ocean_available_out = snakemake.output['ocean_available']
    ocean_missing_out = snakemake.output['ocean_missing']
    #-Script------------------------------------------------------------
    # Concat atmos
    atmos_missing, atmos_available = read_files(atmos_in)
    sum(atmos_missing).to_csv(atmos_missing_out)
    pd.concat(atmos_available).to_csv(atmos_available_out)
    # Concat ocean
    ocean_missing, ocean_available = read_files(ocean_in)
    pd.concat(ocean_missing).to_csv(ocean_missing_out)
    pd.concat(ocean_available).to_csv(ocean_available_out)
    # Concat CMIP
    cmip_missing, cmip_available = read_files(cmip_in)
    pd.concat(cmip_missing).to_csv(cmip_missing_out)
    # For CMIP, subset only files for which all necessary variables
    # have been downloaded.
    cmip_available = (
        pd.DataFrame(
            pd.concat(cmip_available)['cmipfiles']
            .reset_index(drop=True)
            .apply(parsecmip).tolist()
        ).groupby(['model', 'ripf', 'exp'])
         .filter(
             lambda g: {'zos', 'zostoga', 'tos', 'psl',
                        'vas', 'uas'}.issubset(set(g['var']))
        )
    )['fname']
    cmip_available.to_csv(cmip_available_out)


if __name__ == "__main__":
    main()
