"""Process NOAA Tides and Currents data into yearly parquets and CSVs."""

import datetime as dt
from pathlib import Path
from functools import reduce
import pandas as pd
from snakemake.script import snakemake


def msl_monthly(
    fnames: list[str],
    year: int,
    station_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Get monthly MSL for all stations for year."""
    msl_monthly_df = (
        pd.concat(
            pd.read_csv(f).assign(stn=f.split("_")[-2])
            for f in fnames
        ).loc[
            lambda df: [
                int(year)-1 <= x <= int(year)+1
                for x in df['Year']]
        ]
    )
    msl_monthly_df = (
        msl_monthly_df
        .set_index(msl_monthly_df['stn'].astype(int))
        .join(station_metadata
              .set_index('id')[['lat', 'lng']])
        .reset_index(drop=True)
        [["Year", " Month", "stn", "lat", "lng", " Linear_Trend"]]
        .rename({" Month": "Month", " Linear_Trend": "msl"}, axis=1)
    )
    return msl_monthly_df


def twl_hourly(fnames: list[str]) -> pd.DataFrame:
    """Hourly twl for all stations."""
    return (
        pd.concat(
            pd.read_csv(f, parse_dates=[0])
            .assign(stn=f.split("_")[-2])
            for f in fnames
        )
        .rename({'value': 'twl'}, axis=1)
        .set_index('time')
    )


def tides_hourly(fnames: list[str]) -> pd.DataFrame:
    """Hourly tides for all stations."""
    return (
        pd.concat(
            pd.read_csv(f, parse_dates=[0])
            .assign(stn=f.split("_")[-2])
            for f in fnames
        )
        .rename({'value': 'tides'}, axis=1)
        .set_index('time')
    )


def ntr(
    twl_hourly: pd.DataFrame,
    tides_hourly: pd.DataFrame,
    msl_hourly: pd.DataFrame
) -> pd.DataFrame:
    """Calculate NTR as TWL - T - MSL.
    NOTE: MSL here is the linear trend of monthly MSL with seasonality removed.
    All variables are referenced to the recent tidal epoch mean sea level datum.
    """
    df = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            how="left",
            left_on=["time", "stn"],
            right_on=["time", "stn"],
        ),
        [twl_hourly, tides_hourly, msl_hourly],
    ).assign(ntr=lambda df: df["twl"] - df["tides"] - df["msl"])
    return df


def interp(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly to hourly linear interpolation for MSL"""
    df["time"] = [dt.datetime(y, m, 15, 0, 0, 0)
                  for y, m in zip(df.Year, df.Month)]
    return (
        df.reset_index(drop=True)
        .sort_values("time")
        .set_index("time")
        .resample("1h")
        .interpolate(method="linear")
    )


def hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Resample the hourly dataframe to daily."""
    return (
        hourly_df.groupby("stn")
        .apply(
            lambda x: x.sort_values("time")
            .set_index("time")
            .resample(dt.timedelta(days=1))
            .mean(),
        )
        .reset_index()
        .dropna()
    )


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    year = snakemake.params['year']
    end_year = snakemake.params['endyear']
    rawdata_dir = snakemake.params['rawdata_dir']
    hourly_processed_file = snakemake.output['hourly']
    daily_processed_file = snakemake.output['daily']
    monthly_processed_file = snakemake.output['monthly']
    station_metadata = pd.read_csv(snakemake.params['station_metadata_file'])
    fnames_in_raw = [
        rawdata_dir.joinpath(Path(f).name) for f in
        pd.read_csv(snakemake.input['available_files'])['noaa_files'].tolist()
    ]
    #-Script------------------------------------------------------------
    files = [f for f in fnames_in_raw if f[-8:-4] == year]
    msl_raw_files = [f for f in fnames_in_raw if "meantrend" in f]
    tides_raw_files = [f for f in files if "tides" in f]
    twl_raw_files = [f for f in files if "twl" in f]
    if int(year) <= int(end_year):
        # MSL
        msl_monthly_df = msl_monthly(msl_raw_files, year, station_metadata)
        msl_hourly_df = msl_monthly_df.groupby("stn").apply(
            interp, include_groups=False).reset_index()
        # TWL
        twl_hourly_df = twl_hourly(twl_raw_files)
        # Tides
        tides_hourly_df = tides_hourly(tides_raw_files)
        # NTR
        df_hourly = ntr(twl_hourly_df, tides_hourly_df, msl_hourly_df).dropna()
        # combine daily
        df_daily = hourly_to_daily(df_hourly)
        # Write to files
        msl_monthly_df.to_csv(monthly_processed_file)
        df_daily.to_csv(daily_processed_file)
        df_hourly.to_csv(hourly_processed_file)
    else:  # only tides
        tides_hourly_df = tides_hourly(tides_raw_files).reset_index()
        tides_hourly_df = (
            tides_hourly_df
            .set_index(tides_hourly_df['stn'].astype(int))
            .join(station_metadata
                  .set_index('id')[['lat', 'lng']])
            .reset_index(drop=True)
            [["time", "stn", "lat", "lng", "tides"]]
        )
        tides_daily = hourly_to_daily(tides_hourly_df.reset_index())
        tides_hourly_df.to_csv(hourly_processed_file)
        tides_daily.to_csv(daily_processed_file)
        Path(monthly_processed_file).touch()


if __name__ == "__main__":
    main()
