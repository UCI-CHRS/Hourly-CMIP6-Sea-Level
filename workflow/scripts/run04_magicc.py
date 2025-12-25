"""Run MAGICC.
    NOTE: MAGICC only needs to be run once per emissions scenario; it won't
    change between specific ripfs or models.
    PyMAGICC documentation: https://pymagicc.readthedocs.io/en/latest/
"""
from pathlib import Path
import pymagicc
import scmdata
from pymagicc import rcp26, rcp45, rcp60, rcp85
import pandas as pd
from snakemake.script import snakemake
import aux.magicc as mu


def histnat() -> pymagicc.io.MAGICCData:
    """Run MAGICC for the hist-nat emissions scenario. 

    Args: 

    Returns: 
        pymagicc.io.MAGICCData:
            PyMAGICC results object, from the PyMAGICC documentation: 
            "Output of the run with the data in the df attribute and 
            parameters and other metadata in the metadata attribute"
    """
    kwargs = {
        'out_sealevel': True,
        'SLR_SEMIEMPI_SWITCHFROMOBS2CALC': 1850,
        'SLR_SEMIEMPI_TEMPBASEPERIOD': [1850, 1909],
        'startyear': 1850,
        'endyear': 2000,
    }
    infile_patterns = ["*IN", "*.prn"]
    exceptions = ["*RF"]
    with pymagicc.MAGICC6() as magicc:
        rundir = magicc.run_dir
        emission_files = mu.emission_infiles(source_dir=Path(rundir),
                                             infile_patterns=infile_patterns,
                                             exceptions=exceptions)
        for file in emission_files:
            header, body = mu.get_infile_header(file)
            my_infile = mu.infile(magicc=magicc, fname=file)
            df = mu.propagate_baseline_emissions(
                magicc, Path(file), Path(rundir)
            )
        results = magicc.run(
            **kwargs).set_meta(dimension="scenario", value="hist-nat")
    return results


def _run_ssp(ssp: int, rawdata_cmip_sspemissions_path: Path) -> pymagicc.io.MAGICCData:
    """

    Args: 
        scen (pymagicc.io.MAGICCData):
            SSP scenario data object.

    Returns: 
        pymagicc.io.MAGICCData:
            PyMAGICC results object, from the PyMAGICC documentation: 
            "Output of the run with the data in the df attribute and 
            parameters and other metadata in the metadata attribute"
    """
    mu.make_ssp_scen_files(rawdata_cmip_sspemissions_path)
    with pymagicc.MAGICC6() as magicc:
        scen = mu.ssp_scen_files(ssp, rawdata_cmip_sspemissions_path)
        magicc.set_output_variables(sealevel=True)
        # results = magicc.run().set_meta(dimension="scenario", value="historical")
        results = magicc.run(scen).set_meta(
            dimension="scenario", value=f"SSP{ssp}")
    return results


def run_magicc(scenario: str, rawdata_cmip_sspemissions_path: Path) -> pymagicc.io.MAGICCData:
    """Scenarios should match the CMIP6 'experiment' vocabularies. 

    Args: 
        scenario (str):
            name of the scenario

    Returns: 
        pymagicc.io.MAGICCData:
            PyMAGICC results object, from the PyMAGICC documentation: 
            "Output of the run with the data in the df attribute and 
            parameters and other metadata in the metadata attribute"
    """
    match scenario:
        case "hist-nat":
            results = histnat()
        case "historical":
            with pymagicc.MAGICC6() as magicc:
                magicc.set_output_variables(sealevel=True)
                results = magicc.run().set_meta(dimension="scenario", value="historical")
        # SSPs
        case "ssp119":
            results = _run_ssp(119, rawdata_cmip_sspemissions_path)
        case "ssp126":
            results = _run_ssp(126, rawdata_cmip_sspemissions_path)
        case "ssp245":
            results = _run_ssp(245, rawdata_cmip_sspemissions_path)
        case "ssp370":
            results = _run_ssp(370, rawdata_cmip_sspemissions_path)
        case "ssp434":
            results = _run_ssp(434, rawdata_cmip_sspemissions_path)
        case "ssp585":
            results = _run_ssp(585, rawdata_cmip_sspemissions_path)
    return results


def im_te(results: pymagicc.io.MAGICCData | pd.DataFrame, scenario: str
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the thermal expansion and ice melt components at each time step. 

    Args: 
        results (pymagicc.io.MAGICCData): 
            PyMAGICC results object
        scenario (str): 
            scenario name

    Returns: 
        tuple[pd.DataFrame, pd.DataFrame]:
            time series of thermal expansion and ice melt components
    """
    TE_vars = ["SLR_EXPANSION"]
    if isinstance(results, pymagicc.io.MAGICCData):
        results_df = results.filter(
            region="World", scenario=scenario).long_data().set_index("time")
    TE = results_df.loc[results_df['variable'].isin(
        TE_vars)].groupby("time").sum()['value']
    IM = results_df.loc[results_df['variable']
                        == "SLR_SEMIEMPI_TOT", "value"] - TE
    return TE, IM


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs---------------------------------
    magicc_out_dir = Path(snakemake.params['magicc_out_dir'])
    scenarios = snakemake.params['scenarios']
    rawdata_cmip_sspemissions_path = Path(
        snakemake.params['rawdata_cmip_sspemissions_path'])
    # -Script------------------------------------------------------------
    for scen in scenarios:
        fname = magicc_out_dir.joinpath(f"{scen}.csv")
        results = run_magicc(
            scenario=scen,
            rawdata_cmip_sspemissions_path=rawdata_cmip_sspemissions_path
        )
        results.to_csv(fname)


if __name__ == "__main__":
    main()
