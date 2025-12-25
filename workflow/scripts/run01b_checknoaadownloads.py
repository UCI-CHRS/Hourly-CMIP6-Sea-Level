
"""Check for missing NOAA data. """

from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
from snakemake.script import snakemake
from aux import utils


def sum_missing_hourly(
    df: pd.DataFrame
) -> dict[int, dict[float, np.ndarray] | int]:
    """Sum missing values and identify temporal gaps.
    If missing values are not NaN (e.g., -9999),
    they need to be converted first.

    Args:
        df (pd.DataFrame): dataframe with a 'time' and 'value' column

    Returns:
        dict[int, dict[float, np.ndarray] | int]: Dictionary of missing value
            sum and data gaps
    """
    mask = df.index.diff() > df.index.diff().min()
    missing_periods = pd.DataFrame({
        'starts': df.reset_index().shift(1)[mask]['time'],
        'stops':  df.reset_index()[mask]['time'],
    }).assign(duration=lambda df: df['stops'] - df['starts'])
    return missing_periods


def sum_missing_monthly(
    df: pd.DataFrame
) -> dict[int, dict[float, np.ndarray] | int]:
    """Sum missing values and identify temporal gaps in the monthly MSL data.
    If missing values are not NaN (e.g., -9999),
    they need to be converted first.

    Args:
        df (pd.DataFrame): dataframe with a 'Year' and ' Month' column.

    Returns:
        dict[int, dict[float, np.ndarray] | int]: Dictionary of missing value
            sum and data gaps
    """
    timestamps = (
        df.dropna(how='any')
          .assign(time=df['Year'] + df[' Month']/12)['time']
          .to_numpy()
    )
    timestamps_filled_in = np.arange(
        timestamps.min(), timestamps.max() + 1/12, 1/12
    )
    missing_months = list(
        {np.round(x, 3)
         for x in timestamps_filled_in.tolist()
         }.difference({np.round(x, 3) for x in timestamps})
    )
    df = pd.DataFrame({
        'year': [int(np.floor(m)) for m in missing_months],
        'month': [int((m - np.floor(m))*12) for m in missing_months],
        'msl': True,
    }).sort_values(['year', 'month'])
    return df


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    noaa_raw_data_dir = snakemake.params['raw_data_dir']
    stn = snakemake.params['stn']
    sy = int(snakemake.params['sy'])
    ey = int(snakemake.params['ey'])
    missing = snakemake.output['missing']
    available = snakemake.output['available']
    #-Script------------------------------------------------------------
    years = range(sy, ey+1)
    tide_files = list(noaa_raw_data_dir.glob(f"tides_stationid_{stn}_*.csv"))
    twl_files = list(noaa_raw_data_dir.glob(f"twl_stationid_{stn}_*.csv"))
    msl_files = list(noaa_raw_data_dir.glob(
        f"msl_stationid_{stn}_meantrend.csv"))
    if not any(len(f) < 1 for f in (tide_files, twl_files, msl_files)):
        df_tides = pd.concat(
            pd.read_csv(f, parse_dates=[0], index_col=0)
            for f in tide_files
        ).sort_index().rename({'value': 'tides'}, axis=1)
        df_twl = pd.concat(
            pd.read_csv(f, parse_dates=[0], index_col=0)
            for f in twl_files
        ).sort_index().rename({'value': 'twl'}, axis=1)
        df_msl = pd.concat(
            pd.read_csv(f) for f in msl_files
        )
        hourly = df_tides.join(df_twl)[f'{sy}-01-01':f'{ey}-12-31']
        hourly_available = hourly.loc[~hourly.isna().any(axis=1)]
        monthly_available = df_msl.dropna()
        missing_dict = {
            'hourly':  sum_missing_hourly(hourly_available),
            'monthly': sum_missing_monthly(monthly_available),
        }
        df_missing = (
            missing_dict['hourly']
            .join(missing_dict['monthly'], how="outer")
            .reset_index(drop=True)
            .assign(
                msl=lambda df: df['msl'].fillna(value=False),
                station=stn,
                startyear=max(df_twl.index.year.min(),
                              df_tides.index.year.min(), df_msl['Year'].min()),
                endyear=min(df_twl.index.year.max(),
                            df_tides.index.year.max(), df_msl['Year'].max()),
                years=lambda df: df['endyear'] - df['startyear']
            )
        )
        df_missing.to_csv(missing)
        # Report available files
        available_files = pd.DataFrame({
            'noaa_files': [str(f) for f in noaa_raw_data_dir.glob(f"*_stationid_{stn}_*.csv")]
        })
        available_files.to_csv(available)
    else:
        Path(missing).touch()
        Path(available).touch()

if __name__ == "__main__":
    main()
