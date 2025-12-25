"""Download the raw ERA5 atmospheric datasets from the 
Copernicus Climate Data Store (pre-clipped spatially to CONUS)
"""

import os
import datetime as dt
from pathlib import Path
import cdsapi
from snakemake.script import snakemake


def era5h(data_dir: Path, lonmin: float | int, latmin: float | int,
          lonmax: float | int, latmax: float | int, year: int, month: int) -> None:
    """ Use the CCDS API to download one month of the dataset at: 
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
        https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
        clipped to the U.S. Must have .cdsapirc file in home directory with API key. 
        NOTE: this downloads a lot of data to lab server. 
        CDS toolbox tutorial and documentation: 
        https://cds.climate.copernicus.eu/toolbox/doc/how-to/4_how_to_use_output_widgets/4_how_to_use_output_widgets.html#ht4

    Args:
        data_dir (Path): 
            local path to save files to
        lonmin, latmin, lonmax, latmax (float | int): bounding box to clip 
            saved data to
        year (int): 
            Save data for specified 'month' of this year (earliest is 1940, latest is current year)
        month (int): 
            Save data for this month of 'year'

    Returns: 
        None (writes files to disk)
    """
    c = cdsapi.Client()
    fname = data_dir.joinpath(
        f"era5_hourly_wind_slp_{year}_{str(month).zfill(2)}.nc")
    if not fname.exists():
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': ['reanalysis'],
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'mean_sea_level_pressure'  # ,
                    # "10m_u_component_of_neutral_wind",
                    # "10m_v_component_of_neutral_wind",
                    # "ocean_surface_stress_equivalent_10m_neutral_wind_direction",
                    # "ocean_surface_stress_equivalent_10m_neutral_wind_speed"
                ],
                'year': [str(year)],
                'month': [str(month).zfill(2)],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [latmax, lonmin, latmin, lonmax],
                'data_format': 'netcdf',
                'download_format': "unarchived"
            },
            fname)


def era5m(data_dir: Path, lonmin: float | int, latmin: float | int,
          lonmax: float | int, latmax: float | int,
          sy: int = 1940, ey: int = dt.datetime.now().year) -> None:
    """Download monthly SST ERA5 data.

    Args:
        data_dir (Path): 
            local path to save files to
        lonmin, latmin, lonmax, latmax (float | int): bounding box to clip 
            saved data to
        sy (int): 
            beginning year of data record to download (earlies is 1940)
        ey (int): 
            end year of data record to download (latest is current year)

    Returns: 
        None (saves directly to disk)
    """
    fname = data_dir.joinpath("era5_monthly_sst.nc")
    if not fname.exists():
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'sea_surface_temperature',
                'year': [str(y) for y in range(sy, ey + 1)],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'time': '00:00',
                'area': [latmax, lonmin, latmin, lonmax],
            },
            fname)


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    data_dir = snakemake.params['data_dir']
    bbox_conus = snakemake.params['bbox_conus']
    lonmin, latmin, lonmax, latmax = (
        bbox_conus[name] for name in ['xmin', 'ymin', 'xmax', 'ymax']
    )
    # For monthly sst only
    sy = snakemake.params['sy']
    ey = snakemake.params['ey']
    #-Script------------------------------------------------------------
    # Download hourly ERA5 atmospheric observations for current year
    for month in range(1, 13):
        era5h(
            data_dir=data_dir,
            lonmin=lonmin,
            latmin=latmin,
            lonmax=lonmax,
            latmax=latmax,
            year=snakemake.params['year'],
            month=month
        )
    # Download monthly ERA5 atmospheric observations
    # (sea surface temperature only) if it doesn't exist.
    era5m(
        data_dir=data_dir,
        lonmin=lonmin,
        latmin=latmin,
        lonmax=lonmax,
        latmax=latmax,
        sy=sy,
        ey=ey
    )


if __name__ == "__main__":
    main()
