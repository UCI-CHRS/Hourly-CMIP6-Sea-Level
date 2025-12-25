
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np
from snakemake.script import snakemake
from aux import enso


def match_params_by_date(
    df_daily: pd.DataFrame,
    df_params: pd.DataFrame,
    params_to_keep: tuple[str]
) -> pd.DataFrame:
    """Daily parameter lookup."""
    return pd.merge(
        left=df_daily.assign(
            mmdd=df_daily.index.strftime("%m%d")).reset_index(),
        right=(
            df_params.loc[
                [x in params_to_keep for x in df_params.param]
            ].pivot(columns="param", values="value")
            .assign(mmdd=lambda df: df.index.strftime("%m%d"))
            .reset_index(drop=True)
        ),
        left_on="mmdd",
        right_on="mmdd"
    ).set_index("time")


def ntr_lowfreq(
    obs_enso: pd.DataFrame,
    tos: xr.Dataset,
    param_regression_lf: pd.DataFrame
) -> pd.DataFrame:
    """ NTR: Calculate ENSO3.4 from TOS layer and apply ENSO regression"""
    enso34_cmip = pd.DataFrame({
        'nino34': enso.detrended_enso_34_sst(obs_enso['nino34'], tos)
    })
    enso_34_dailyinterp = enso.interpolate_sst(
        enso34_cmip, sst_col="nino34"
    ).set_index("time")
    data_with_params = match_params_by_date(
        enso_34_dailyinterp, param_regression_lf, ('m', 'b')
    )
    ntr_lf = (
        data_with_params['m'] *
        data_with_params['nino34'] +
        data_with_params['b']
    )
    ntr_lf.index += dt.timedelta(hours=12)
    return pd.DataFrame({'ntr_lf': ntr_lf})


def _bc(df_daily, diff, ratio, variable):
    df = df_daily[[variable, 'dayofyear']]
    mean_diff = diff[f'{variable}_mean'].to_dataframe().reset_index()
    std_ratio = ratio[f'{variable}_std'].to_dataframe().reset_index()
    # Get the mean diff and std ratio for each day of the year in df_daily
    df = pd.merge(
        pd.merge(df, mean_diff, left_on="dayofyear", right_on="dayofyear"),
        std_ratio, left_on="dayofyear", right_on="dayofyear"
    )
    # Shift by mean_diff
    shift = df[variable] - df[f'{variable}_mean']
    # Divide by std ratio
    bc = shift / df[f'{variable}_std']
    bc.index = df_daily.index
    df_daily[f'{variable}_bc'] = bc
    return df_daily


def bias_correct(df_daily, bc_coord):
    diff = (
        bc_coord.sel(stat_source='hist') -
        bc_coord.sel(stat_source='obs')
    )
    ratio = (
        bc_coord.sel(stat_source='hist') /
        bc_coord.sel(stat_source='obs')
    )
    df_daily['dayofyear'] = df_daily.index.dayofyear.values
    for variable in ['uas', 'vas', 'psl']:
        df_daily = _bc(df_daily, diff, ratio, variable)
    return df_daily


def standardize(df_with_params, var, var_bc):
    x = df_with_params[var_bc]
    std = df_with_params[f'z_std_{var}']
    mu = df_with_params[f'z_mean_{var}']
    x_stan = (x - mu)/std
    return x_stan


def check_extrap(df, var, min_var, max_var):
    extrap = (
        (df[var] < df[min_var]) &
        (df[var] > df[max_var])
    )
    return extrap


def ntr_highfreq(
    df_daily: pd.DataFrame,
    param_regression_hf: pd.DataFrame,
    bc_coord: xr.Dataset,
) -> pd.DataFrame:
    """ NTR: Apply slp/wind regression"""
    if df_daily is None:
        return None
    if any(x not in df_daily.columns for x in ('psl', 'uas', 'vas')):
        return None
    # Bias correct
    df_daily = bias_correct(df_daily, bc_coord)
    # Calculate pseudo wind stress
    df_daily['tauu'] = np.multiply(np.multiply(
        np.sign(df_daily.uas_bc), df_daily.uas), df_daily.uas_bc)
    df_daily['tauv'] = np.multiply(np.multiply(
        np.sign(df_daily.vas_bc), df_daily.vas_bc), df_daily.vas_bc)
    # Match parameters for each day
    df_with_params = match_params_by_date(
        df_daily,
        param_regression_hf,
        ('m_tauu', 'm_tauv', 'm_slp', 'b',
         'z_mean_tauu', 'z_std_tauu', 'min_tauu', 'max_tauu',
         'z_mean_tauv', 'z_std_tauv', 'min_tauv', 'max_tauv',
         'z_mean_slp', 'z_std_slp', 'min_slp', 'max_slp')
    )
    # Flag extrapolated values
    extrap = {
        'tauu_extrap':
            check_extrap(
                df_with_params, 'tauu', 'min_tauu', 'max_tauu'
            ),
        'tauv_extrap':
            check_extrap(
                df_with_params, 'tauv', 'min_tauv', 'max_tauv'
            ),
        'slp_extrap':
            check_extrap(
                df_with_params, 'psl_bc', 'min_slp', 'max_slp'
            )
    }
    # Standardize data
    df_with_params['tauu_stan'] = standardize(df_with_params, 'tauu', 'tauu')
    df_with_params['tauv_stan'] = standardize(df_with_params, 'tauv', 'tauv')
    df_with_params['slp_stan'] = standardize(df_with_params, 'slp', 'psl_bc')
    # Apply regression relationship and flag extrapolated values
    ntr_hf = pd.DataFrame({
        'ntr_hf':
            df_with_params['m_tauu'] * df_with_params['tauu_stan'] +
            df_with_params['m_tauu'] * df_with_params['tauv_stan'] +
            df_with_params['m_slp'] * df_with_params['slp_stan'] +
            df_with_params['b'],
        'extrap_hf':
            extrap['tauu_extrap'] | extrap['tauv_extrap'] | extrap['slp_extrap']
    })
    return ntr_hf


def main(
    bias_correction: xr.Dataset,
    regression_daily_params: pd.DataFrame,
    regression_monthly_params: pd.DataFrame,
    cmip_daily: xr.Dataset,
    cmip_sst: xr.Dataset,
    lat: float,
    lon: float,
    ntr_daily_file: str,
    obs_enso: pd.DataFrame,


):
    """Main script to run via snakemake."""
    cmip_daily_df = (
        cmip_daily.sel(x=lon, y=lat, method="nearest")
        .to_dataframe().drop(['x', 'y'], axis=1)
    )
    # Daily NTR
    bc_coord = bias_correction.sel(x=lon, y=lat, method="nearest")
    ntr_hf = ntr_highfreq(
        cmip_daily_df, regression_daily_params, bc_coord)
    ntr_lf = ntr_lowfreq(
        obs_enso, cmip_sst['tos'], regression_monthly_params
    )
    ntr_d = ntr_hf.join(ntr_lf).dropna()
    ntr_d['ntr_total'] = ntr_d['ntr_hf'] + ntr_d['ntr_lf']
    ntr_d.to_csv(ntr_daily_file)


def parse_args():
    args = dict(
        # -Snakemake params, inputs, outputs----------------------------
        bias_correction=xr.open_dataset(snakemake.input['bias_correction']),
        regression_daily_params=pd.read_csv(
            snakemake.input['regression_daily'],
            index_col="ts",
            parse_dates=True,
            usecols=['ts', 'param', 'value']
        ),
        regression_monthly_params=pd.read_csv(
            snakemake.input['regression_monthly'],
            index_col="ts",
            parse_dates=True,
            usecols=['ts', 'param', 'value']
        ),
        cmip_daily=xr.open_dataset(snakemake.input['cmip_daily']),
        cmip_sst=xr.open_dataset(snakemake.input['cmip_sst']),
        lat=float(snakemake.wildcards['lat']),
        lon=float(snakemake.wildcards['lon']),
        ntr_daily_file=snakemake.output[0],
        obs_enso=pd.read_csv(snakemake.input['enso']),
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
