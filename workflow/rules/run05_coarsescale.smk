

### --------------------------------------------------------------------
### Get bias correction parameters (additive mean, multiplicative st dev shifts)
### --------------------------------------------------------------------


rule biascorrect:
    input:
        upsample = f"{BASEDIR}/00_metadata/touchfiles/upsample.done",
        cmip = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/cmip/"
            "{model}_{ripf}_historical_daily.nc"
        )
    output:
        (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "biascorr/{model}_{ripf}.nc"
        )
    params:
        upsampled_atmos_daily = Path(
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_atmos_{SRC_CONFIG['obs_atmos_source']}"
        ),
        glob_pattern_daily_atmos = f"{SRC_CONFIG['obs_atmos_source']}_daily_*.nc",
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear'],
        ndays = GRD_CONFIG['ndays_window'],
        ngrids = GRD_CONFIG['ngrids_window'],

    script:
        "../scripts/run05a_biascorrect.py"


rule calculatedatums:
    input:
        (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/cmip/"
            "{model}_{ripf}_historical_monthly.nc"
        )
    output: 
        (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/historical_monthly_datums.nc"
        )
    run:
        from pathlib import Path
        import xarray as xr
        Path(output[0]).parents[0].mkdir(exist_ok=True)
        Path(output[0]).parents[0].joinpath("hourly").mkdir(exist_ok=True)
        zos_zostoga = xr.open_dataset(input[0])
        zos_zostoga_datums = (
            zos_zostoga
            .sel(time=slice("1983-01-01", "2001-12-30"))
            .mean("time", skipna=True)
        )
        zos_zostoga_datums.to_netcdf(output[0])


rule monthlymsl:
    input:
        datums = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/historical_monthly_datums.nc"
        ),
        magicc = f"{BASEDIR}/04_params/magicc/{{exp}}.csv",
        cmip_monthly = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/cmip/"
            "{model}_{ripf}_{exp}_monthly.nc"
        )
    output:
        f"{BASEDIR}/05_cmip/{config['grid_params']}_"
        f"{config['obs_source_params']}_{config['dipmac_params']}/"
        "{model}_{ripf}_{exp}/msl_monthly.nc"
    script:
        "../scripts/run05b_monthlymsl.py"


rule dailyntr:
    input:
        bias_correction = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "biascorr/{model}_{ripf}.nc"
        ),
        regression_daily = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "lat_{lat}_lon_{lon}_ntr_daily_regression.csv"
        ),
        regression_monthly = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "lat_{lat}_lon_{lon}_ntr_monthly_regression.csv"
        ),
        cmip_daily = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            "cmip/{model}_{ripf}_{exp}_daily.nc"
        ),
        cmip_sst = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            "cmip/{model}_{ripf}_{exp}_enso.nc"
        ),
        enso = f"{BASEDIR}/01_rawdata/obs/ocean/noaa/nino34.csv"
    output:
        (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_daily_lat_{lat}_lon_{lon}.csv"
        )
    script:
        "../scripts/run05c_dailyntr.py"


rule dailyntrcmipdists:
    input:
        (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_daily_lat_{lat}_lon_{lon}.csv"
        )
    output: 
        (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_daily_dists_marg_{marg}_lat_{lat}_lon_{lon}.csv"
        )
    params:
        ndays = GRD_CONFIG['ndays_window'],
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear']
    script:
        "../scripts/run05d_dailyntrcmip_dists.py"
