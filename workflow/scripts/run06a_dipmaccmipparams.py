"""Get the hourly marginal params for each CMIP run.
"""


from pathlib import Path
import os
import numpy as np
import pandas as pd
from snakemake.script import snakemake
from aux import stats as s


def multiplicative_shift(hourly_obs, daily_obs, daily_cmip):
    """Use for scale parameters."""
    return hourly_obs * daily_cmip / daily_obs


def additive_shift(hourly_obs, daily_obs, daily_cmip):
    """Use for location and transformed shape parameters."""
    return hourly_obs + daily_cmip - daily_obs


def transformed_additive_shift(oh, od, cd, tfunc, tfunc_inv, limits):
    transformed_shift = additive_shift(tfunc(oh), tfunc(od), tfunc(cd))
    shift = tfunc_inv(transformed_shift)
    shift_bounded = np.clip(shift, *limits)
    return shift_bounded


def param_limits(marginal_family, param):
    match param, marginal_family:
        case 'a', 'skewnorm':
            limits = [-50, 50]
        case 'df', 't':
            limits = [2.1, 200]
        case _:
            limits = [-np.inf, np.inf]
    return limits


def compute_loc(oh, od, cd, marginal_family):
    return additive_shift(oh, od, cd)


def compute_scale(oh, od, cd, marginal_family):
    return multiplicative_shift(oh, od, cd)


def compute_a(oh, od, cd, marginal_family):
    if marginal_family == "skewnorm":
        tfunc = np.arcsinh
        tfunc_inv = np.sinh
    else:
        raise Exception(
            "(run06a_dipmaccmipparams.compute_a) "
            "Transformation function "
            f"not specified for marginal_family={marginal_family}."
        )
    limits = param_limits(marginal_family, 'a')
    shifted = transformed_additive_shift(oh, od, cd, tfunc, tfunc_inv, limits)
    return shifted


def compute_df(oh, od, cd, marginal_family):
    if marginal_family == "t":
        tfunc = np.log
        tfunc_inv = np.exp
    else:
        raise Exception(
            "(run06a_dipmaccmipparams.compute_d) "
            "Transformation function "
            f"not specified for marginal_family={marginal_family}."
        )
    limits = param_limits(marginal_family, 'df')
    shifted = transformed_additive_shift(oh, od, cd, tfunc, tfunc_inv, limits)
    return shifted


def clip_params(df, value_col, marginal_family):
    param_list = df['param'].unique()
    limits = {p: param_limits(marginal_family, p) for p in param_list}
    lower = df["param"].map(lambda p: limits[p][0])
    upper = df["param"].map(lambda p: limits[p][1])
    return df[value_col].clip(lower=lower, upper=upper)


def apply_correct_shifts(cmip_params_df, marginal_family):
    df = cmip_params_df.reset_index().set_index('ts').assign(hourly_cmip=np.nan)
    # Limit param values to ranges in param_limits()
    df_clip = df.assign(
        hourly_obs=clip_params(df, 'hourly_obs', marginal_family),
        daily_obs=clip_params(df, 'daily_obs', marginal_family),
        daily_cmip=clip_params(df, 'daily_cmip', marginal_family),
    )
    # Apply shifts
    rules = {
        "loc":   compute_loc,
        "scale": compute_scale,
        "a":     compute_a,
        "df":     compute_df,
    }
    df_clip["hourly_cmip"] = [
        rules[p](oh, od, cd, marginal_family)
        for p, oh, od, cd in 
        zip(
            df_clip["param"],
            df_clip["hourly_obs"],
            df_clip["daily_obs"],
            df_clip["daily_cmip"]
        )
    ]
    return df_clip


def getparams(df, param_names):
    df_mod = df.loc[[x in param_names for x in df['param']]]
    return df_mod.reset_index().set_index(['ts', 'param'])['value']


def shift_marg_params(
    marginal_hourly_obs_params: pd.DataFrame,
    marginal_daily_obs_params: pd.DataFrame,
    marginal_daily_cmip_params: pd.DataFrame,
    param_names: list[str],
    marginal_family: str
) -> pd.DataFrame:
    """Estimate emissions-based shift in the hourly distribution."""
    cmip_params = pd.DataFrame({
        'hourly_obs': getparams(marginal_hourly_obs_params, param_names),
        'daily_obs': getparams(marginal_daily_obs_params, param_names),
        'daily_cmip': getparams(marginal_daily_cmip_params, param_names)
    })
    cmip_params_shift = apply_correct_shifts(cmip_params, marginal_family)
    cmip_params_shift_round = np.round(cmip_params_shift, 4)
    return cmip_params_shift_round


def read_parameter_csvs(fname: str):
    return pd.read_csv(
        fname,
        parse_dates=['ts'],
        usecols=['ts', 'param', 'value']
    )


def main(
    marg_params_hourly_obs: str,
    marg_params_daily_obs: str,
    marg_params_daily_cmip: str,
    marginal_family: str,
    output_cmip_params_csv: str
):
    """Main script to run via snakemake."""
    if os.stat(marg_params_daily_cmip).st_size > 0:
        # Get transformed CMIP marginal distribution parameters
        param_names = s.distribution_param_names(marginal_family)
        cmip_params = shift_marg_params(
            read_parameter_csvs(marg_params_hourly_obs),
            read_parameter_csvs(marg_params_daily_obs),
            read_parameter_csvs(marg_params_daily_cmip),
            param_names,
            marginal_family
        )
        cmip_params.to_csv(output_cmip_params_csv)
    else:
        Path(output_cmip_params_csv).touch()


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        marg_params_hourly_obs = snakemake.input['marg_params_hourly_obs'],
        marg_params_daily_obs = snakemake.input['marg_params_daily_obs'],
        marg_params_daily_cmip = snakemake.input['marg_params_daily_cmip'],
        marginal_family = snakemake.wildcards['margfamily'],
        output_cmip_params_csv = snakemake.output['output_cmip_params_csv']
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
