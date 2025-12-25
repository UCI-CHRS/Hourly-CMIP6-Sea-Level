
"""Script to retrieve NOAA TWL, MSL, and tide data. See API documentation at:
    https://api.tidesandcurrents.noaa.gov/api/prod/
    https://api.tidesandcurrents.noaa.gov/mdapi/prod/
    https://api.tidesandcurrents.noaa.gov/dpapi/prod/
Datum information: 
    https://tidesandcurrents.noaa.gov/datum_options.html
NOTE: The ENSO SST data is not saved here, but read from source in src.msl.enso_regression. 
"""
from pathlib import Path
import datetime as dt
import os
from dataclasses import dataclass
import requests
import pandas as pd
from snakemake.script import snakemake


def one_month_noaa_data(
    product: str, application: str, sd: dt.datetime, ed: dt.datetime,
    stn: int, info_key: str, interval: str | None = None,
) -> pd.DataFrame | None:
    """Downloads one month of data from the NOAA API, since
    only 31 days of data can be requested at a time.
    Assumes: GMT time zone, metric units, JSON format
    API docs: 
        https://api.tidesandcurrents.noaa.gov/api/prod/
        https://api.tidesandcurrents.noaa.gov/mdapi/prod/
        https://api.tidesandcurrents.noaa.gov/dpapi/prod/
    Datum information: 
        https://tidesandcurrents.noaa.gov/datum_options.html

    Args: 
        product (str): 
            NOAA product to query; see API docs
        application (str): 
            NOAA application to query; see API docs. Likely "NOS.COOPS.TAC.WL"
        sd (dt.datetime):
            start date of request
        ed (dt.datetime):
            end date of request
        stn (str):
            Station ID to query
        info_key (str): 
            inf_key value
        interval (str | None): 
            temporal resolution of data to retrieve, e.g., "h" for hourly. 

    Returns: 
        pd.DataFrame | None: 
            requested data table
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"  # base url

    def params(sd, ed):
        return {
            'product': product,
            'application': application,
            'begin_date': sd.strftime("%Y%m%d"),
            'end_date': ed.strftime("%Y%m%d"),
            'datum': "MSL",
            'station': stn,
            'time_zone': "GMT",  # greenwich mean time
            'units': "metric",
            'interval': interval,
            'format': "json",
        }
    try:
        r = requests.get(url, params(sd, ed))
    except requests.exceptions.ReadTimeout:
        return None
    if r.status_code == 200:
        try:
            info = r.json()
            if not 'error' in info.keys():
                df = pd.DataFrame(info[info_key])
            else:
                df = None
        except requests.exceptions.JSONDecodeError:
            df = None
    else:
        df = None
    return df


def month_end(sd: dt.datetime) -> tuple[dt.datetime]:
    """Returns the start and end date of a specific month. 

    Args: 
        sd (dt.datetime): start of month (should be the first)

    Returns: 
        dt.datetime: end of the month
    """
    return sd + pd.DateOffset(months=1) - pd.DateOffset(days=1)


def one_year_noaa_data(year, one_month_noaa_data_kwargs):
    """Loop API requests monthly, since only 31 days of data can be 
        requested at a time. 

    Args: 
        startyear (int): first year of the decade
        one_month_noaa_data_kwargs (dict[str, str]): dictionary of 
            args to pass to one_month_noaa_data()

    Returns: 
        df (pd.DataFrame): decade of concatenated monthly datasets
    """
    start_dates = [dt.date(year, 1, 1) + pd.DateOffset(months=m)
                   for m in range(12)]
    end_dates = [month_end(sd) for sd in start_dates]
    df_list = [one_month_noaa_data(
        sd=sd, ed=ed, **one_month_noaa_data_kwargs)
        for sd, ed in zip(start_dates, end_dates)
    ]
    if len([x for x in df_list if x is not None]) > 0:
        df_raw = pd.concat(df_list).rename(
            {'t': 'time', 'v': 'value'}, axis=1).set_index("time")
        df_raw.index = pd.to_datetime(df_raw.index)
        df = df_raw.loc[df_raw['value'].str.len() > 0, 'value'].astype(float)
        return df


def tides(rawdir: Path, stn: str, year: int) -> None:
    """Download astronomical tide data from the NOAA T&C API 
    for one station for one year. 

    Args:
        rawdir (Path): 
            directory to save data to
        stn (str):
            station ID to query
        year (int): 
            year of data to download

    Returns: 
        None (saves directly to disk)
    """
    fname = rawdir.joinpath(
        f"tides_stationid_{stn}_{year}.csv")
    if not fname.exists():
        df = one_year_noaa_data(
            year=year,
            one_month_noaa_data_kwargs=dict(
                product="predictions",
                application="NOS.COOPS.TAC.WL",
                stn=stn,
                info_key="predictions",
                interval="h"
            )
        )
        if df is not None:
            df.to_csv(fname)


def twl(rawdir: str, stn: str, year: int) -> None:
    """Download total water level data from the NOAA T&C API 
    for one station for one year

    Args:
        rawdir (Path): 
            directory to save data to
        stn (str):
            station ID to query
        year (int): 
            year of data to download

    Returns: 
        None (saves directly to disk)
    """
    fname = rawdir.joinpath(
        f"twl_stationid_{stn}_{year}.csv")
    if not fname.exists():
        df = one_year_noaa_data(
            year=year,
            one_month_noaa_data_kwargs=dict(
                stn=stn,
                product="hourly_height",
                application="NOS.COOPS.TAC.WL",
                info_key="data"
            )
        )
        if df is not None:
            df.to_csv(fname)


def msl_monthly(rawdir: Path, stn: str) -> None:
    """Monthly sea level values. Per data source, 
        "values are relative to the most recent Mean Sea Level datum established by CO-OPS."
        Thus, this data has the same datum as anything generated from the
        noaa_api dataclass, which uses the "MSL" datum. 
        NOTE: this has seasonality removed, so the monthly seasonality
        for each station is also downloaded.

    Args:
        rawdir (Path): 
            directory to save data to
        stn (str):
            station ID to query

    Returns: 
        None (saves directly to disk)
    """
    # Monthly MSL with seasonal trend removed
    fname_monthly_msl = rawdir.joinpath(f"msl_stationid_{stn}_meantrend.csv")
    if not fname_monthly_msl.exists():
        url = f"https://tidesandcurrents.noaa.gov/sltrends/data/{stn}_meantrend.csv"
        try:
            response = requests.get(url)
        except requests.exceptions.ReadTimeout:
            pass
        else:
            if response.status_code == 200:
                df = pd.read_csv(url, header=4, index_col=False)
                df.to_csv(fname_monthly_msl, index=False)
    # The seasonal means removed from the monthly MSL data
    fname_seasonal_cycle = rawdir.joinpath(
        f"msl_seasonal_stationid_{stn}.csv")
    if not fname_seasonal_cycle.exists():
        url = f"https://tidesandcurrents.noaa.gov/sltrends/data/{stn}_seasonal.csv"
        try:
            response = requests.get(url)
        except requests.exceptions.ReadTimeout:
            pass
        else:
            if response.status_code == 200:
                df = pd.read_csv(url, header=4, index_col=False)
                df['stn'] = stn
                df.to_csv(fname_seasonal_cycle, index=False)


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    noaa_raw_data_dir = snakemake.params['raw_data_dir']
    stn = snakemake.params['stn']
    sy = int(snakemake.params['sy'])
    ey = int(snakemake.params['ey'])
    #-Script------------------------------------------------------------
    for year in range(sy, 2101):
        # Tides - includes projections up to 2100
        tides(noaa_raw_data_dir, stn=stn, year=year)
        # Historical observations only
        if year <= ey:
            # TWL
            twl(noaa_raw_data_dir, stn=stn, year=year)
            # Monthly MSL - All saved at once; only re-runs if missing.
            msl_monthly(noaa_raw_data_dir, stn=stn)
   

if __name__ == "__main__":
    main()
