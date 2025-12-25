"""Save the station metdata (IDs, lat, lon coordinates) for the 
NOAA Tides and Currents tide gauges. 
"""

from pathlib import Path
import requests
import pandas as pd
from snakemake.script import snakemake


def get_all_noaa_tide_stations():
    """Retrieve all NOAA tide station metadata.

    Args: 

    Returns: 
        pd.DataFrame: 
            tide station metadata
    """
    r = requests.get(
        "https://api.tidesandcurrents.noaa.gov/mdapi/prod/"
        "webapi/stations.json?type=waterlevels"
    )
    stn_dict = r.json()['stations']
    stn_df = pd.DataFrame(stn_dict)
    tide_stations = stn_df.loc[stn_df.tidal]
    station_metadata = tide_stations[[
        'id', 'lat', 'lng', 'state', 'timezonecorr']]
    return station_metadata


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    outfile = Path(snakemake.output[0])
    #-Script------------------------------------------------------------
    get_all_noaa_tide_stations().to_csv(outfile)

if __name__ == "__main__":
    main()
