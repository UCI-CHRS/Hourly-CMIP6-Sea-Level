
"""Combine NTR, tides, MSL into one hourly dataset."""

rule copy_ncs:
    input:
        f"{BASEDIR}/00_metadata/run06makesealevel.done",
        regression_hf = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            "ntr_daily_regression.nc"
        ),
        regression_lf = (
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
        dipmac_marg_hourly = expand(
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "ntr_hourly_dipmac_marg_params_{margfamily}.nc",
            margfamily=config['comboparams']['marginalfamilies']
        ),
    output:
        regression_hf = f"{BASEDIR}/06_final/04_params/ntr_daily_hf_regression.nc",
        regression_lf = f"{BASEDIR}/06_final/04_params/ntr_daily_lf_regression.nc",
        dipmac_marg_hourly = expand(
            f"{BASEDIR}/06_final/04_params/ntr_hourly_dipmac_marg_params_{{marg}}.nc",
            marg=config['comboparams']['marginalfamilies']
        ),
        dipmac_acs = expand(
            f"{BASEDIR}/06_final/04_params/ntr_hourly_dipmac_acs_params_{{acs}}.nc",
            acs=config['comboparams']['acsfamilies']
        )
    params:
        paramdir = Path(f"{BASEDIR}/06_final/params")
    run:
        from pathlib import Path
        import shutil
        # Copy over param .nc files
        param_dir = params['paramdir']
        def copy_and_rename(oldpath: Path, newpath: Path, oldname: str, newname: str):
            shutil.copy(oldpath.joinpath(oldname), newpath.joinpath(newname))
        copy_and_rename(
            Path(input['regression_hf']).parent, param_dir,
            Path(input['regression_hf']).name, "ntr_daily_hf_regression.nc"
        )
        copy_and_rename(
            Path(input['regression_lf']).parent, param_dir,
            Path(input['regression_lf']).name, "ntr_daily_lf_regression.nc"
        )
        for f in input['dipmac_marg_hourly'] + input['dipmac_acs']:
            copy_and_rename(Path(f).parent, param_dir, Path(f).name, Path(f).name)


rule copy_all_tides:
    input:
        # f"{BASEDIR}/00_metadata/run06makesealevel.done"
        f"{BASEDIR}/00_metadata/touchfiles/upsample.done"
    output:
        tides = f"{BASEDIR}/06_final/data/tides.nc"
    params:
        tide_path = (
            f"{BASEDIR}/03_upsampleddata/{config['grid_params']}/"
            f"obs_ocean_{SRC_CONFIG['obs_ocean_source']}"
        )
    run:
        from pathlib import Path
        import pandas as pd
        import xarray as xr
        tide_files = list(Path(params['tide_path']).glob(f"*_hourly_*.nc"))
        tide_df_list = pd.concat([
            (xr.open_dataset(f)['tides']
               .to_dataframe()
               .dropna()
            )
            for f in tide_files
        ], dim="time")
        tides = tides.to_xarray()
        tides.to_netcdf(output['tides'])


rule combine_ntr:
    """Combine NTR csvs into one .nc. Trigger copy_ncs before running this rule."""
    input:
        f"{BASEDIR}/00_metadata/run06makesealevel.done",
        regression_hf = f"{BASEDIR}/06_final/04_params/ntr_daily_hf_regression.nc",
        regression_lf = f"{BASEDIR}/06_final/04_params/ntr_daily_lf_regression.nc",
        dipmac_marg_hourhly = expand(
            f"{BASEDIR}/06_final/04_params/ntr_hourly_dipmac_marg_params_{{marg}}.nc",
            marg=config['comboparams']['marginalfamilies']
        ),
        dipmac_acs = expand(
            f"{BASEDIR}/06_final/04_params/ntr_hourly_dipmac_acs_params_{{acs}}.nc",
            acs=config['comboparams']['acsfamilies']
        )
    params:
        ntr_csv_dir = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
        ),
    output:
        ntr = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_ntr.nc"
        )
    script:
        "../scripts/run07a_combinentr.py"


rule get_tides:
    input:
        # f"{BASEDIR}/00_metadata/run06makesealevel.done",
        tides = f"{BASEDIR}/06_final/data/tides.nc",
        ntr = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_ntr.nc"
        )
    output:
        tides = f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_tides.nc"
    run:
        from pathlib import Path
        import pandas as pd
        import xarray as xr
        ntr = xr.open_dataset(input['ntr'], chunks={'time': 100})
        x_vals, y_vals, t_vals = ntr.x, ntr.y, ntr.time
        tides = xr.open_dataset(input['tides'])
        tides['time'] = pd.to_datetime(tides['time'])
        tides = tides.sel(x=x_vals, y=y_vals, time=t_vals)
        tides.attrs = {
            'tides': "Hourly astronomical tides."
        }
        tides = tides.compute()
        tides.to_netcdf(output['tides'])


rule interpolate_msl:
    input:
        # f"{BASEDIR}/00_metadata/run06makesealevel.done",
        msl_monthly = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/msl_monthly.nc"
        ),
        ntr = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_ntr.nc"
        )
    output:
        msl_hourly = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_msl.nc"
        )
    script:
        "../scripts/run07b_interpolatemsl.py"


rule compiletwl:
    """Input is the makesealevel touchfile"""
    input:
        tides_hourly = f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_tides.nc",
        msl_hourly = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_msl.nc"
        ),
        ntr_hourly = (
            f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_ntr.nc"
        )
    output:
        f"{BASEDIR}/06_final/data/{{model}}_{{ripf}}_{{exp}}_{{marg}}_{{acs}}_twl.nc"
    params:
        data_dir = lambda wc: Path(
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            f"{wc.model}_{wc.ripf}_{wc.exp}"
        )
    script:
        "../scripts/run07c_compiletwl.py"
