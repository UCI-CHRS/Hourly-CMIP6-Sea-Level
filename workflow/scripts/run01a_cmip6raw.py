"""Script to download CMIP6 data."""

from pathlib import Path
import os
import subprocess
import yaml
import requests
import pandas as pd
import pyesgf
from pyesgf.search import SearchConnection
from snakemake.script import snakemake


def parse_cmip_ids(id: str, sep: str = ".") -> dict[str, str]:
    """Returns named properties of a CMIP run by parsing its ID.

    Args:
        id (str):
            standard CMIP6 ID string
        sep (str):
            string separating ID components

    Returns:
        dict[str, str]:
            named CMIP ID components
    """
    if "|" in id:
        id = id.split("|")[0]
    if len(id.split(".")) == 9:
        mip_era, activity, institution_id, source_id, experiment_id, \
            member_id, table_id, variable_id, grid_label = id.split(sep)
    elif len(id.split(".")) == 10:
        mip_era, activity, institution_id, source_id, experiment_id, \
            member_id, table_id, variable_id, grid_label, version = id.split(
                sep)
    else:
        raise Exception("Unknown ID format.")
    metadata = {
        'mip_era': mip_era, 'activity': activity, 'institution_id': institution_id,
        'source_id': source_id, 'experiment_id': experiment_id, 'member_id': member_id,
        'table_id': table_id, 'variable_id': variable_id, 'grid_label': grid_label
    }
    return metadata


def pyesgf_download_one_result(result: pyesgf.search.results.DatasetResult) -> str:
    """Process one result from the pyesgf search interface. 

    Args:
        result (pyesgf.search.results.DatasetResult): search result

    Returns: 
        dict: 
            log of successful and unsuccessful file writes within the result
    """
    result_id = result.json['id'].split("|")[1]
    result_log = {f"{result_id}_search": True}
    try:
        files = result.file_context().search()
    except (
        requests.exceptions.SSLError,
        pyesgf.search.exceptions.EsgfSearchException,
        requests.exceptions.HTTPError
    ) as exc:
        print(exc)
        files = []
        result_log[f"{result_id}_search"] = False
        return result_log
    for file in files:
        fname = file.filename
        url = file.download_url
        result_log[fname] = True
        if not Path(fname).exists():
            try:
                subprocess.run(
                    ["wget", url],
                    check=True,
                    stdout=subprocess.PIPE
                )
            except subprocess.CalledProcessError as grepexc:
                print("error code", grepexc.returncode, grepexc.output)
                result_log[fname] = False
    return result_log


def pyesgf_download(master_id: str,
                    replica: bool = False,) -> None:
    """
    from pyesgf.search import SearchConnection. 
    NOTE: the from_timestamp and to_timestamp arguments in the search 
    context don't seem to work for CMIP6 searches. 
    NOTE: files are downloaded to the current working directory
    """
    # Suppress the facets warning, since facets are specified
    os.environ['ESGF_PYCLIENT_NO_FACETS_STAR_WARNING'] = "true"
    facets = 'mip_era,source_id,experiment_id,member_id,table_id,variable_id'
    conn = SearchConnection(
        'https://esgf.ceda.ac.uk/esg-search',
        distrib=True,
    )
    print(f"Running master ID {master_id}")
    metadata = parse_cmip_ids(master_id)
    logs = dict(mid_search=True)
    try:
        ctx = conn.new_context(
            facets=facets,
            mip_era=metadata['mip_era'],
            source_id=metadata['source_id'],
            experiment_id=metadata['experiment_id'],
            member_id=metadata['member_id'],
            table_id=metadata['table_id'],
            variable_id=metadata['variable_id'],
            replica=replica,
            latest=True,
        )
        results = ctx.search(query=f"id:{master_id.split('|')[0][:-10]}*")
    except (
        requests.exceptions.SSLError,
        requests.exceptions.HTTPError
    ) as exc:
        print(exc)
        results = []
        logs['mid_search'] = False
        return logs
    print(f"Number of results: {len(results)}")
    result_logs = [pyesgf_download_one_result(result) for result in results]
    return logs | {k: v for d in result_logs for k, v in d.items()}


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    mid = snakemake.params['mid']
    ncdir = snakemake.params['ncdir']
    logfile = snakemake.output['logfile']
    #-Script------------------------------------------------------------
    project_dir = os.getcwd()
    os.chdir(ncdir)
    mid_logs = pyesgf_download(mid, replica=True)
    os.chdir(project_dir)
    # Write out logs
    with open(logfile, "w", encoding="utf-8") as f:
        yaml.dump(mid_logs, f)

if __name__ == "__main__":
    main()
