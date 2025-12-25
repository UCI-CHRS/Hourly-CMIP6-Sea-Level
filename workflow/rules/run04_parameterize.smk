
### --------------------------------------------------------------------
### MAGICC 
### --------------------------------------------------------------------

rule magicc:
    """
    Run in docker (or singularity) container because MAGICC requires
    wine to run on linux and mac.
    Example:
    container:
        docker://registry.gitlab.com/phd4943231/cmip6-hourly-sea-level
    To run locally:
    cd workflow/envs && docker build .
    docker run -it -d <IMAGE_ID> /bin/bash
    docker exec -it <CONTAINER_ID> /bin/bash
    NOTE: The HOME_DIRECTORY and BASE_DATA_PATH in config.yml should 
    reflect the home directory /home/myuser and whereever the base
    data path has been bind-mounted within the container when running
    this rule. 
    """
    input: 
        ocean = f"{BASEDIR}/00_metadata/ocean_metadata_manifest.csv",
        cmip = f"{BASEDIR}/00_metadata/cmip_metadata_manifest.csv"
    output: 
        [
            f"{BASEDIR}/04_params/magicc/{scen}.csv"
            for scen in EMISSION_SCENARIOS
        ]
    params:
        scenarios = tuple(EMISSION_SCENARIOS),
        magicc_out_dir = f"{BASEDIR}/04_params/magicc",
        rawdata_cmip_sspemissions_path = f"{BASEDIR}/01_rawdata/cmip/ssp_emissions",
    script:
        "../scripts/run04_magicc.py"

### --------------------------------------------------------------------
### DiPMaC parameter fits
### --------------------------------------------------------------------

rule dipmacfitmarg:
    """Fit the marginal distribution for one location (lat, lon)
    Resolution is daily or hourly.
    """
    input:  # needs upsampled ocean obs (hourly)
        f"{BASEDIR}/00_metadata/touchfiles/upsample.done"
    output:
        (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "lat_{lat}_lon_{lon}_ntr_{resolution}_dipmac_marg_params_{margfamily}.csv"
        )
    params:
        upsampled_ocean_dir = Path(
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}"
        ),
        glob_pattern = lambda wc: (
            f"{SRC_CONFIG['obs_ocean_source']}_{wc.resolution}_*.nc"
        ),
        ref_grid = f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc",
        xmin = BBOX['xmin'],
        ymin = BBOX['ymin'],
        xmax = BBOX['xmax'],
        ymax = BBOX['ymax'],
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear'],
        ndays = GRD_CONFIG['ndays_window'],
        ngrids = GRD_CONFIG['ngrids_window'],
        margfamily = lambda wc: wc.margfamily,
        lat = lambda wc: float(wc.lat),
        lon = lambda wc: float(wc.lon)
    script:
        "../scripts/run04a_dipmac_fitmarg.py"


## Rule for one location (lat, lon)
rule dipmacfitacs:
    input:  # needs upsampled ocean obs (hourly)
        f"{BASEDIR}/00_metadata/touchfiles/upsample.done"
    output:
        (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "lat_{lat}_lon_{lon}_ntr_hourly_dipmac_acs_params_{acsfamily}.csv"
        )
    params:
        upsampled_ocean_dir = Path(
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}"
        ),
        glob_pattern = lambda wc: (
            f"{SRC_CONFIG['obs_ocean_source']}_hourly_*.nc"
        ),
        ref_grid = f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc",
        xmin = BBOX['xmin'],
        ymin = BBOX['ymin'],
        xmax = BBOX['xmax'],
        ymax = BBOX['ymax'],
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear'],
        ndays = GRD_CONFIG['ndays_window'],
        ngrids = GRD_CONFIG['ngrids_window'],
        acsfamily = lambda wc: wc.acsfamily,
        lagmax = DIPMAC_CONFIG['lagmax'],
        lat = lambda wc: float(wc.lat),
        lon = lambda wc: float(wc.lon)
    script:
        "../scripts/run04a_dipmac_fitacs.py"

### --------------------------------------------------------------------
### Regression parameter fits
### --------------------------------------------------------------------

rule getenso:
    """Save NINO3.4 ENSO data from NOAA."""
    output:
        f"{BASEDIR}/01_rawdata/obs/ocean/noaa/nino34.csv"
    run:
        enso = (
            pd.read_fwf(
                "https://www.cpc.ncep.noaa.gov/data/indices/"
                "ersst5.nino.mth.91-20.ascii"
            )[["YR", "MON", "ANOM.3"]]
            .rename(
                {"YR": "year", "MON": "month", "ANOM.3": "nino34"},
                axis=1
            ).set_index(["year", "month"]).squeeze()
        )
        enso.to_csv(output[0])


rule regressionfit:
    input:  # needs upsampled ocean and atmos obs (daily)
        upsample = f"{BASEDIR}/00_metadata/touchfiles/upsample.done",
        enso = f"{BASEDIR}/01_rawdata/obs/ocean/noaa/nino34.csv"
    output:
        daily = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "lat_{lat}_lon_{lon}_ntr_daily_regression.csv"
        ),
        monthly = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "lat_{lat}_lon_{lon}_ntr_monthly_regression.csv"
        )
    params:
        upsampled_ocean_daily = Path(
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}"
        ),
        glob_pattern_daily_ocean = f"{SRC_CONFIG['obs_ocean_source']}_daily_*.nc",
        upsampled_atmos_daily = Path(
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_atmos_{SRC_CONFIG['obs_atmos_source']}"
        ),
        glob_pattern_daily_atmos = f"{SRC_CONFIG['obs_atmos_source']}_daily_*.nc",
        ref_grid = f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc",
        xmin = BBOX['xmin'],
        ymin = BBOX['ymin'],
        xmax = BBOX['xmax'],
        ymax = BBOX['ymax'],
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear'],
        gridname = GRD_CONFIG['gridname'],
        ndays = GRD_CONFIG['ndays_window'],
        ngrids = GRD_CONFIG['ngrids_window'],
        lat = lambda wc: float(wc.lat),
        lon = lambda wc: float(wc.lon)
    script:
        "../scripts/run04a_regressionfit.py"

### --------------------------------------------------------------------
### Combine parameter CSVs into .nc files
### --------------------------------------------------------------------

def param_latlons():
    # checkpoint_output = checkpoints.upsample.get().output[0]
    checkpoint_output = f"{BASEDIR}/00_metadata/upsampled_data_manifest.csv"
    df = pd.read_csv(checkpoint_output)
    lats, lons = df['y'], df['x']
    return lats, lons


def combineparamncs_dipmac_input(wildcards):
    lats, lons = param_latlons()
    return [
        f"{BASEDIR}/04_params/dipmac_regression/"
        f"{config['grid_params']}/{config['obs_source_params']}/"
        f"{config['dipmac_params']}/"
        f"lat_{lat}_lon_{lon}_ntr_{wildcards.temporalres}_"
        f"dipmac_{wildcards.paramtype}_"
        f"{wildcards.param_or_actpnts}_{wildcards.distfamily}.csv"
        for lat, lon in zip(lats, lons)
    ]

rule combineparamncs_dipmac:
    """Write combined param ncs.
    Expand lats and lons for one dipmac family combination based on grid 
    (see *grid_stations*.csv in upsampleddata/grids)
    """
    input:
        combineparamncs_dipmac_input
    output:
        (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "ntr_{temporalres}_dipmac_{paramtype}_{param_or_actpnts}_{distfamily}.nc"
        )
    params:
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear']
    script:
        "../scripts/run04b_combineparamncs.py"


def combineparamncs_regression_input(wildcards):
    lats, lons = param_latlons()
    return [
        f"{BASEDIR}/04_params/dipmac_regression/"
        f"{config['grid_params']}/{config['obs_source_params']}/"
        f"lat_{lat}_lon_{lon}_ntr_{wildcards.temporalres}_regression.csv"
        for lat, lon in zip(lats, lons)
    ]

rule combineparamncs_regression:
    """Write combined regression param ncs.
    Expands lats and lons for one dipmac family combination based on grid 
    (see *grid_stations*.csv in upsampleddata/grids)
    """
    input: 
        combineparamncs_regression_input
    output: 
        (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "ntr_{temporalres}_regression.nc"
        )
    params:
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear']
    script:
        "../scripts/run04b_combineparamncs.py"


### --------------------------------------------------------------------
### Params checkpoint.
### --------------------------------------------------------------------

checkpoint params:
    input:
        magicc = [
            f"{BASEDIR}/04_params/magicc/{scen}.csv"
            for scen in EMISSION_SCENARIOS
        ],
        regression_daily = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "ntr_daily_regression.nc"
        ),
        regression_monthly = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "ntr_monthly_regression.nc"
        ),
        dipmac_acs = expand(
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "ntr_hourly_dipmac_acs_params_{acsfamily}.nc",
            acsfamily=config['comboparams']['acsfamilies']
        ),
        dipmac_marg_daily = expand(
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "ntr_daily_dipmac_marg_params_{margfamily}.nc",
            margfamily=config['comboparams']['marginalfamilies']
        ),
        dipmac_marg_hourly = expand(
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "ntr_hourly_dipmac_marg_params_{margfamily}.nc",
            margfamily=config['comboparams']['marginalfamilies']
        ),
        upsample_checkpoint = f"{BASEDIR}/00_metadata/upsampled_data_manifest.csv"
    output:
        f"{BASEDIR}/00_metadata/param_available_latlons_manifest.csv"
    params:
        paramdir_regression = Path(
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
        )
    run:
        import os
        from pathlib import Path
        import pandas as pd
        latlon_files = [
            Path(f) for f in params['paramdir_regression'].glob("lat_*lon_*.csv")
            if os.stat(f).st_size > 0
        ]
        latlons = list({
            (float(str(f.name).split("_")[1]),  # lat
            float(str(f.name).split("_")[3]))  # lon
            for f in latlon_files
        })
        df_paramlatlons = pd.DataFrame(latlons, columns=("lat", "lon"))
        df_upsample = pd.read_csv(input["upsample_checkpoint"], index_col=0)
        df = pd.merge(
            left=df_paramlatlons,
            right=df_upsample,
            how="inner",
            left_on=['lon', 'lat'],
            right_on=['x', 'y']
        )
        df.to_csv(output[0])
