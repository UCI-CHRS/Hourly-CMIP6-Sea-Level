"""Retrieve CMIP6 GCM metadata. 
Includes set of functions for searching and returning CMIP6 datasets using the
ESGF search API. API docs at: 
https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API
"""

from pathlib import Path
import itertools
import requests
import pandas as pd
from snakemake.script import snakemake
from aux import utils


class UrlStatusException(Exception):
    """Exception raised for requests.get output without status code 200"""

    def __init__(self, request):
        self.message = (f"Status code {request.status_code} returned. "
                        "URL not working as expected.")
        super().__init__(self.message)


def check_url_status(r: requests.models.Response) -> None:
    """Raises an exception if anything other than status 200 is raised
        for request 'r'.

    Args:
        r (requests.models.Response):
            requests object from URL

    Returns:
        None (raises UrlStatusException or does nothing)
    """
    if r.status_code == 200:
        pass
    else:
        raise UrlStatusException(r)


def cmip_search_url(my_query: str | None = None, **my_params) -> tuple[str, dict]:
    """Generates a CMIP6 data search url.
        For url, use cmip_search_url().url
        For search results, use cmip_search_url().json()
        or return_search_results_dataframe() wrapper.
        API docs: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Args:
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at:
            https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        tuple[str, dict]:
            url and params to pass to a requests module HTTP request.
    """
    url = "https://esgf-node.llnl.gov/esg-search/search"
    url = "https://esgf.ceda.ac.uk/esg-search/search"
    params = {'limit': 10000,  # Per API docs, system imposes max limit <= 10,000
              'latest': "true",
              'format': "application/solr+json",
              'project': "CMIP6",
              #   'replica': "true",
              'shards': ("esgf-index1.ceda.ac.uk/solr,"  # search all nodes
                         #  "esgf-node.llnl.gov/solr,"
                         "esgf-node.ipsl.upmc.fr/solr,"
                         "esgf-data.dkrz.de/solr")
              }
    if my_query is not None:
        params['query'] = my_query
    for key, value in my_params.items():
        if isinstance(value, list):
            params[key] = ",".join(value)
        else:
            params[key] = value
    return url, params


@utils.timing
def return_cmip_search(search_url: str | None = None, my_query: str | None = None,
                       **my_params) -> list:
    """API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination
        Adapted from esgf_search() function at: https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0
        To search opendap urls, use param: files_type="OPENDAP"

    Args:
        search_url (str):
            url to pass to requests.get(). If none, search url will be
            constructed from my_query and **my_params
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        list:
            list of CMIP search results.
    """
    offset = 0
    numFound = 10000
    client = requests.session()
    result_list = []
    while offset < numFound:
        if offset > 0:
            print(f"Processing result {offset} out of {numFound}")
        if search_url is None:
            url, params = cmip_search_url(my_query, offset=offset, **my_params)
            r = client.get(url=url, params=params)
        else:
            r = client.get(url=search_url, params={"offset": offset})
        check_url_status(r)
        resp = r.json().get('response')
        numFound = int(resp["numFound"])
        results = resp.get('docs')
        result_list.append(results)
        offset += len(results)
    return utils.flatten_list(result_list)


def cmip_search_opendap_urls(my_query: str | None = None, **my_params) -> list[str]:
    """Returns OPENDAP urls for a search. Note that usable URLs aren't
        always available.

    Args:
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        list[str]:
            list of OPENDAP urls from a search.
    """
    res = return_cmip_search(my_query, **my_params)
    opendap_urls = []
    files_type = "OPENDAP"
    for d in res:
        if "url" in d.keys():
            for f in d["url"]:
                # opendap_urls.append(f)
                sp = f.split("|")
                if sp[-1] == files_type:
                    opendap_urls.append(sp[0].split(".html"[0]))
    return opendap_urls


def return_search_results_dataframe(my_query: str | None = None, **my_params) -> pd.DataFrame:
    """Returns CMIP6 search results as a pandas dataframe.

    Args:
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        pd.DataFrame:
            table of search results from the json dictionary returned by the HTTP request.
    """
    search_results_dict = return_cmip_search(my_query=my_query, **my_params)
    df = pd.DataFrame(search_results_dict)
    return df


def model_info_df(my_query: str | None = None, **my_params) -> pd.DataFrame:
    """DataFrame of the model info (with some variables removed).

    Args:
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        pd.DataFrame:
            table of search results.
    """
    df = return_search_results_dataframe(my_query, **my_params)
    cols_to_keep = ['id', 'activity_id',
                    'cf_standard_name', 'experiment_id', 'experiment_title',
                    'frequency', 'grid', 'grid_label', 'institution_id',
                    'master_id', 'nominal_resolution', 'source_id',
                    'sub_experiment_id', 'table_id', 'variable_id', 'variable_long_name',
                    'variable_units', 'variant_label', 'realm']
    model_info = df[cols_to_keep].reset_index().apply(pd.Series.explode)
    # model_info['id'] = [x.split("|")[0] for x in model_info.id]
    return model_info


def and_search(and_variables: list[str], groupers: list[str],
               my_query: str | None = None, **my_params) -> pd.DataFrame:
    """Returns dataset groups containing all specified and_variables.

    Args:
        and_variables (list[str]):
            list of variable names that must be available in all returned models
        groupers (list[str]):
            list of parameters that define one model run.
            Likely ['source_id', 'variant_label']
        my_query (str | None):
            query string to pass to the ESGF search API
        **my_params:
            key word arguments to pass to the ESGF search API. See all
            keyword options at: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API

    Returns:
        pd.DataFrame:
            subset table of search results with the condition that every
            and_variable is available for every model.
    """
    or_search_results = model_info_df(my_query, **my_params)

    def subset(and_var, and_var_values, grouped):
        def enforce_and(myvars):
            return lambda x: all([(x == a).any() for a in myvars])
        return grouped[and_var].transform(enforce_and(and_var_values))
    df = or_search_results.copy()
    df['and_combos'] = list(zip(*[df[c] for c in and_variables]))
    and_var_values = [my_params[and_var] for and_var in and_variables]
    and_combos = list(itertools.product(*and_var_values))
    grouped = df.groupby(groupers)
    and_subset_idx = subset('and_combos', and_combos, grouped)
    and_subset_df = or_search_results.loc[and_subset_idx]
    return and_subset_df


def return_consistent_models(vars: list[str],
                             experiments: list[str],
                             **and_search_kwargs) -> pd.DataFrame:
    """Returns datasets for all modeling groups that have the hist-nat
        run for each of the specified 'vars' variable_ids.

    Args:
        vars (list[str]):
            list of variable_ids
        experiments (list[str]):
            list of experiment_ids
        **and_search_kwArgs:
            key word arguments to pass to and_search()

    Returns:
        pd.DataFrame:
            dataframe subsetted to an and_search on and_variables
            "experiment_id" and "variable_id" and groupers "source_id"
            and "variant_label"
    """
    and_subset_df = and_search(
        and_variables=['experiment_id', 'variable_id'],
        groupers=['source_id', 'variant_label'],
        experiment_id=experiments,
        variable_id=vars,
        **and_search_kwargs)
    return and_subset_df


def cmip_vars() -> tuple[list[str], list[str]]:
    """Dummy function to store the ocean and atmospheric CMIP6 variable names to get.

    Args:

    Returns:
        tuple[list[str], list[str]]:
            ocean and atmospheric variable names
    """
    ocean_vars = ['zos', 'zostoga']
    # tauu, tauv for wind stress
    atmospheric_vars = ['psl', 'uas', 'vas', 'tos']
    return ocean_vars, atmospheric_vars


def slr_cmip_models(experiments: list[str]) -> pd.DataFrame:
    """Returns a table of CMIP6 models filtered based on the gcm class attributes
    that have both the ocean and atmospheric variables available.

    Args:
        experiments (list[str]): list of grouped experiments 

    Returns:
        pd.DataFrame:
            table of CMIP6 models filtered based on the gcm class attributes
    """
    identifiers = ['source_id', 'variant_label']
    ocean_vars, atmospheric_vars = cmip_vars()
    atmospheric_df = return_consistent_models(
        atmospheric_vars, experiments, frequency="day"
    )
    ocean_df = return_consistent_models(
        ocean_vars, experiments, frequency="mon")
    atmospheric_models = set(
        list(zip(*[atmospheric_df[c] for c in identifiers]))
    )
    ocean_models = set(list(zip(*[ocean_df[c] for c in identifiers])))
    common_models = atmospheric_models & ocean_models  # check overlap

    ocean_df["identifier"] = list(zip(*[ocean_df[c] for c in identifiers]))
    atmospheric_df["identifier"] = list(
        zip(*[atmospheric_df[c] for c in identifiers]))
    usable_datasets = pd.concat(
        [atmospheric_df.loc[atmospheric_df["identifier"].isin(common_models)],
            ocean_df.loc[ocean_df["identifier"].isin(common_models)]],
        axis=0
    )
    return usable_datasets


def grid_metadata(grid: str) -> str:
    """Shorten the grid metadata.

    Args:
        grid (str):
            long gridname

    Returns:
        str:
            shortened gridname
    """
    grid_shorten = (grid.lower().replace("native", "n")
                                .replace("data regridded to a", "m")
                                .replace(";", "")
                                .replace("(", "")
                                .replace(")", "")
                                .replace("n96 grid", "n96")
                                .replace("longitude/latitude", "ll")
                                .replace("ocean tri-polar grid", "ocean_tripolar")
                                .replace("atmosphere", "atmos")
                                .replace("cmip6 standard 1x1 degree lonxlat grid", "stan1x1")
                                .replace("t127l reduced gaussian grid", "t127_rgg")
                                .replace("meridional refinement down to 1/3 degree in the tropics", "1_3_deg_tropics")
                                .replace("using an area-average preserving method", "aapm")
                                .replace(" ", "_"))
    return grid_shorten[:80]


def remove_500km(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 500km (CanESM5) and duplicate entries with different grids.
        Specifically remove gr1 grids first.
        Removes in place

    Args:
        df (pd.DataFrame): table of CMIP run metadata 
            (as produced by slr_cmip_models)

    Returns:
        pd.DataFrame:
            subset slr_cmip_models without 500km resolution models.
    """
    df['ripf'] = [x[x.index("i"):] for x in df['variant_label']]
    df = df.drop_duplicates(subset=['id']).reset_index()
    df = df.loc[(df.source_id != "CanESM5") & (df.grid_label != "gr1")]
    df['unq'] = list(zip(*[df[c] for c in ["source_id", "variant_label",
                                           "experiment_id", "variable_id"]]))
    df.drop_duplicates(subset="unq", inplace=True)
    return df


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs---------------------------------
    experiments = snakemake.params['experiments']
    metadata_folder = snakemake.params['metadata_folder']
    # -Script------------------------------------------------------------
    for run_name, run_experiments in experiments.items():
        # Retrieve GCM metadata
        df = slr_cmip_models(run_experiments).pipe(remove_500km)
        df['grid_id'] = [grid_metadata(grid) for grid in df['grid']]
        df.to_csv(
            metadata_folder.joinpath(f"{run_name}.csv"),
            index=False
        )


if __name__ == "__main__":
    main()
