
"""Global sea level change time series data:
    https://cds.climate.copernicus.eu/datasets/sis-water-level-change-timeseries-cmip6
Includes TWL, MSL, tides, and storm surge.
"""

import shutil
import os
import pathlib
import cdsapi
import numpy as np
from snakemake.script import snakemake


def gtsm(ddir: pathlib.Path) -> None:
    """Downloads Global Surge and Tide Model data from the Copernicus
        Climate Data Store, including:
            Annual mean sea level (MSL) (available 1950 - 2015)
            10-minute tides (available 1950 - 2015)
            Hourly storm surge (available 1979 - 2015)
            Hourly total water level (available 1979 - 2015)
        NOTE: TWL and storm surge available from 1979 - 2015, tides and MSL from 1950.
        All data saved 1979-2015 for consistency.

    Args:
        ddir (pathlib.Path):
            local directory in which to save data (rawdata_obs_ocean_gstm)

    Returns:
        None (saves data directly to disk)
    """
    # TWL and storm surge available from 1979, tides and MSL from 1950
    # years = np.arange(1979, 2015)
    years = np.arange(1950, 2050)
    c = cdsapi.Client()
    os.chdir(ddir)

    def nums_to_str(nums: list[int], length: int) -> list[str]:
        """Takes a range of numbers and returns a list of zero-padded
            strings.

        Args:
            nums (list[int]):
                list of integers to zero-pad
            length (int):
                length of the zero-padded strings.

        Returns:
            list[str]:
                zero-padded string representation of each of nums
        """
        return [str(x).zfill(length) for x in nums]

    # MSL - annual
    fname = ddir.joinpath("msl.zip")
    if not fname.exists():
        c.retrieve(
            'sis-water-level-change-timeseries-cmip6',
            {
                'format': 'zip',
                'variable': 'mean_sea_level',
                'experiment': 'historical',
                'temporal_aggregation': 'annual',
                'year': nums_to_str(range(years[0], years[-1]), length=4),
            },
            str(fname))
        shutil.unpack_archive(str(fname), ddir)

    # Retrieve subdaily data 1 year at a time.

    # Tides (10 minute)
    for y1, y2 in zip(years[:-1], years[1:]):
        fname = ddir.joinpath(f"tides_{y1}_{y2}.zip")
        if not fname.exists():
            c.retrieve(
                'sis-water-level-change-timeseries-cmip6',
                {
                    'format': 'zip',
                    'variable': 'tidal_elevation',
                    'experiment': 'historical',
                    'temporal_aggregation': '10_min',
                    'year': nums_to_str(range(y1, y2+1), length=4),
                    'month': nums_to_str(range(1, 13), length=2),
                },
                str(fname))
        shutil.unpack_archive(str(fname), ddir)

        # Storm surge (hourly)
        fname = ddir.joinpath(f"surge_{y1}_{y2}.zip")
        if not fname.exists():
            c.retrieve(
                'sis-water-level-change-timeseries-cmip6',
                {
                    'format': 'zip',
                    'variable': 'storm_surge_residual',
                    'experiment': 'reanalysis',
                    'temporal_aggregation': 'hourly',
                    'year': nums_to_str(range(y1, y2+1), length=4),
                    'month': nums_to_str(range(1, 13), length=2),
                },
                str(fname))
            shutil.unpack_archive(str(fname), ddir)

        # Total water level (hourly)
        fname = ddir.joinpath(f"twl_{y1}_{y2}.zip")
        if not fname.exists():
            c.retrieve(
                'sis-water-level-change-timeseries-cmip6',
                {
                    'format': 'zip',
                    'variable': 'total_water_level',
                    'experiment': 'reanalysis',
                    'temporal_aggregation': 'hourly',
                    'year': nums_to_str(range(y1, y2+1), length=4),
                    'month': nums_to_str(range(1, 13), length=2),
                },
                str(fname))
            shutil.unpack_archive(str(fname), ddir)


def main():
    """Main script to run via snakemake.
    NOTE: This is untested in current form.
    """
    #-Snakemake params, inputs, outputs---------------------------------
    gtsm_raw_data_dir = snakemake.params['raw_data_dir']
    startyear = int(snakemake.params['year'])
    ey = int(snakemake.params['ey'])
    #-Script------------------------------------------------------------
    gtsm(gtsm_raw_data_dir)


if __name__ == "__main__":
    main()
