
"""
## These should be run on an HPC, and have high memory requirements. 
## Use snakemake directly with snakemake --profile workflow/profiles/slurm -R rule
## (but can also just be run as a slurm job array)
## See docs at https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
"""


rule upsample_atmos:
    input:
        processed_atmos = (
            f"{BASEDIR}/02_processeddata/obs_atmos_{SRC_CONFIG['obs_atmos_source']}/"
            f"{SRC_CONFIG['obs_atmos_source']}_daily_{{year}}.nc"
        ),
        ref_grid = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        )
    output:
        (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/obs_atmos_{SRC_CONFIG['obs_atmos_source']}/"
            f"{SRC_CONFIG['obs_atmos_source']}_daily_{{year}}.nc"
        )
    params:
        gridname = GRD_CONFIG['gridname'],
        year = lambda wc: wc.year
    script: 
        f"../scripts/run03_{SRC_CONFIG['obs_atmos_source']}_upsample.py"


rule upsample_ocean:
    input:
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
        ),
        ref_grid = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        )
    output:
        upsampled_hourly = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_hourly_{{year}}.nc"
        ),
        upsampled_daily = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_daily_{{year}}.nc"
        ),
        upsampled_monthly = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
            f"{SRC_CONFIG['obs_ocean_source']}_monthly_{{year}}.nc"
        )
    params: 
        gridname = GRD_CONFIG['gridname'],
        year = lambda wc: wc.year,
        endyear = SRC_CONFIG['endyear']
    script:
        f"../scripts/run03_{SRC_CONFIG['obs_ocean_source']}_upsample.py"


rule upsample_cmip:
    input: 
        zos = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_zos.nc"
        ),
        zostoga = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_zostoga.nc"
        ),
        uas = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_uas.nc"
        ),
        vas = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_vas.nc"
        ),
        psl = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_psl.nc"
        ),
        ref_grid = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        )
    output:
        daily = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"cmip/{{model}}_{{ripf}}_{{exp}}_daily.nc"
        ),
        monthly = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"cmip/{{model}}_{{ripf}}_{{exp}}_monthly.nc"
        )
    script:
        "../scripts/run03_cmip_upsample.py"


rule upsample_cmip_enso:
    input: 
        tos = (
            f"{BASEDIR}/02_processeddata/cmip/{{model}}_{{ripf}}_{{exp}}_tos.nc"
        ),
        ref_grid = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        )
    output:
        enso = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"cmip/{{model}}_{{ripf}}_{{exp}}_enso.nc"
        )
    run:
        import xarray as xr
        from aux import geo
        ds_cmip = xr.open_dataset(input['tos'])
        ref_grid = xr.open_dataset(input['ref_grid'])
        ds_upsampled = geo.regrid_to_match(
            ds_src=ds_cmip,
            ds_match=ref_grid,
            src_x="x",
            src_y="y",
            match_x="x",
            match_y="y",
        )
        ds_upsampled.to_netcdf(output['enso'])


def upsample_input_atmos():
    years = range(SRC_CONFIG['startyear'], SRC_CONFIG['endyear']+1)
    return [
        f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
        f"obs_atmos_{SRC_CONFIG['obs_atmos_source']}/"
        f"{SRC_CONFIG['obs_atmos_source']}_daily_{year}.nc"
        for year in years
    ]


def upsample_input_ocean():
    hourly = [
        f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
        f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
        f"{SRC_CONFIG['obs_ocean_source']}_hourly_{year}.nc"
        for year in range(SRC_CONFIG['startyear'], 2101)
    ]
    daily = [
        f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
        f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
        f"{SRC_CONFIG['obs_ocean_source']}_daily_{year}.nc"
        for year in range(SRC_CONFIG['startyear'], SRC_CONFIG['endyear']+1)
    ]
    monthly = [
        f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
        f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}/"
        f"{SRC_CONFIG['obs_ocean_source']}_monthly_{year}.nc"
        for year in range(SRC_CONFIG['startyear'], SRC_CONFIG['endyear']+1)
    ]
    return hourly + daily + monthly

def upsample_input_cmip(_):
    checkpoint_output = checkpoints.rawdata.get().output['cmip_available']
    files = pd.read_csv(checkpoint_output)['fname']
    mids = pd.DataFrame(dict(
        model=[f.split("_")[2] for f in files],
        ripf=[f.split("_")[4] for f in files],
        exp=[f.split("_")[3] for f in files]
    ))

    return [
        f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/cmip/"
        f"{model}_{ripf}_{exp}_{variable}.nc"
        for model, ripf, exp in zip(mids['model'], mids['ripf'], mids['exp'])
        for variable in ['daily', 'monthly', 'enso']
    ]

checkpoint upsample:
    """NOTE: input is based on the rawdata files."""
    input:
        atmos = upsample_input_atmos(),
        ocean = upsample_input_ocean(),
        cmip = upsample_input_cmip,
        ref_grid_latlons = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}_"
            f"ngrids_{GRD_CONFIG['ngrids_window']}_noaastns.csv"
        )
    output:
        f"{BASEDIR}/00_metadata/upsampled_data_manifest.csv"
    params:
        upsampled_cmip_dir = Path(f"{BASEDIR}/03_upsampleddata/grd001/cmip"),
    run:
        import pandas as pd
        upsampled_cmip_files = pd.DataFrame(
            {'file': list(params['upsampled_cmip_dir'].glob("*.nc"))}
        )
        latlons = pd.read_csv(input['ref_grid_latlons'], index_col=0)
        upsampled_cmip_files['key'] = 1
        latlons['key'] = 1
        df = upsampled_cmip_files.merge(latlons, on="key").drop(columns="key")
        df.to_csv(output[0])
