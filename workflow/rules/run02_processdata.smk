
"""Process raw atmos and ocean data. 
## These rules should be run on an HPC. Use snakemake directly with 
## snakemake --profile workflow/profiles/slurm -R rule
## (but can also just be run as a slurm job array)
## See docs at https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
"""


rule process_atmos:
    input:
        # process_atmos_input
        available_files = f"{BASEDIR}/00_metadata/atmos_rawdata_manifest.csv"
    output:
        # processed daily atmos file
        daily = (
            f"{BASEDIR}/02_processeddata/obs_atmos_{SRC_CONFIG['obs_atmos_source']}/"
            f"{SRC_CONFIG['obs_atmos_source']}_daily_{{year}}.nc"
        )
    params:
        year = lambda wc: wc.year,
        rawdata_dir = Path(f"{BASEDIR}/01_rawdata/obs/atmos/{SRC_CONFIG['obs_atmos_source']}")
    script:
        f"../scripts/run02_atmos_{SRC_CONFIG['obs_atmos_source']}_process.py"


rule process_ocean:
    input: 
        available_files = f"{BASEDIR}/00_metadata/ocean_rawdata_manifest.csv"
    output:
        hourly = (
            f"{BASEDIR}/02_processeddata/obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_hourly_{{year}}.csv"
        ),
        daily = (
            f"{BASEDIR}/02_processeddata/obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_daily_{{year}}.csv"
        ),
        monthly = (
            f"{BASEDIR}/02_processeddata/obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_monthly_{{year}}.csv"
        )
    params:
        year = lambda wc: wc.year,
        endyear = SRC_CONFIG['endyear'],
        station_metadata_file = (
            f"{BASEDIR}/00_metadata/{SRC_CONFIG['obs_ocean_source']}_station_metadata.csv"
        ),
        rawdata_dir = Path(f"{BASEDIR}/01_rawdata/obs/ocean/{SRC_CONFIG['obs_ocean_source']}")
    script:
       f"../scripts/run02_ocean_{SRC_CONFIG['obs_ocean_source']}_process.py"


rule process_cmip:
    """NOTE: We want to get CMIP for the CONUS; all other bboxes
    are a subset of that and don't warrant re-processing.
    SST requires the NINO regions instead. 
    """
    input:
        available_files = f"{BASEDIR}/00_metadata/cmip_rawdata_manifest.csv"
    output:
        f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_{{variable}}.nc"
    params:
        model = lambda wc: wc.model,
        ripf = lambda wc: wc.ripf,
        exp = lambda wc: wc.exp,
        variable = lambda wc: wc.variable,
        cmip_nc_dir = Path(f"{BASEDIR}/01_rawdata/cmip/nc"),
        bbox_conus = config['metadata_keys']['conus'],
        bbox_enso = config['metadata_keys']['nino34']
    log:
        "testpath/{model}_{ripf}_{exp}_{variable}.log"
    script:
       f"../scripts/run02_cmip_process.py"
