
"""Mean sea level calculations + datum correction.
    Each component should be relative to the tidal epoch MSL (1983-2001 values).
    NOTE: all MSL values are monthly.
"""

import pandas as pd
import xarray as xr
import numpy as np
from snakemake.script import snakemake


def get_ice_melt(
    magicc: pd.DataFrame,
    exp: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read MAGICC ice melt and thermal expansion.

    Args:
        magicc (pd.DataFrame): dataframe of MAGICC results
        exp (str): CMIP experiment (e.g., ssp585, hist-nat, historical)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            thermal expansion and ice melt components of global yearly
            mean sea level change for experiment exp.
    """
    if "ssp" in exp:
        exp = exp.upper()
    df = (
        magicc.loc[
            (magicc.region == "World") & (magicc.scenario == exp)
        ]
        .melt(id_vars=[
            'climate_model',
            'model',
            'region',
            'scenario',
            'todo',
            'unit',
            'variable'
        ], var_name="time"
        )
    )
    df['time'] = pd.to_datetime(df['time'])
    thermal_expansion = (
        df.loc[df['variable'] == "SLR_EXPANSION"]
        .groupby("time")
        .sum()['value']
    )
    ice_melt = (
        df.loc[df['variable'] == "SLR_SEMIEMPI_TOT"]
        .groupby("time")
        .sum()['value'] - thermal_expansion
    )
    return thermal_expansion, ice_melt


def get_ice_melt_plus_zostoga(
    zostoga: xr.DataArray,
    te: pd.DataFrame,
    im: pd.DataFrame
) -> pd.DataFrame:
    """Gets the sum of thermal expansion (zostoga) plus ice melt,
    based on the te:im ratio from the MAGICC results
    NOTE: The minimum values are temporarily subtracted in order to
        properly apply the te:im ratio (regardless of datum), and then
        added back in to conserve mass.

    Args:
        zostoga (xr.Dataset):
            CMIP6 thermal expansion time series (relative to datum)
        te (pd.Series):
            global thermal expansion time series from MAGICC (relative to datum)
        im (pd.Series):
            global ice melt time series from MAGICC (relative to datum)

    Returns:
        xr.Dataset:
            sum of thermal expansion and ice melt components
    """
    # Make all values positive for ratio calculation
    minval = min(
        zostoga.values.min(),
        np.nanmin(te),
        np.nanmin(im)
    )
    zostoga_p = zostoga - minval
    te_p = te - minval
    im_p = im - minval
    # Apply yearly global IM:TE ratio to monthly global zostoga
    ratio = pd.merge(
        left=pd.DataFrame({'ratio': im_p/te_p}
                          ).assign(year=lambda df: df.index.year),
        right=(
            zostoga_p
            .to_dataframe()
            .assign(
                year=lambda df: df.index.get_level_values("time").year,
                month=lambda df: df.index.get_level_values("time").month
            )
            .reset_index()
        ),
        left_on="year",
        right_on="year",
        how="inner"
    )
    # Estimate ice melt from zostoga and the IM:TE ratio
    ratio['ice_melt'] = (ratio['zostoga'] * ratio['ratio']) + minval
    ratio['zostoga_plus_ice_melt'] = (
        ratio['zostoga'] +
        ratio['ice_melt'] +
        minval
    )
    return (
        ratio
        .set_index([*zostoga.dims])
        .to_xarray()
        [['zostoga', 'ice_melt', 'zostoga_plus_ice_melt']]
    )


def main():
    """Main script to run via snakemake."""
    # -Snakemake params, inputs, outputs--------------------------------
    datums = xr.open_dataset(snakemake.input['datums'])
    magicc = pd.read_csv(snakemake.input['magicc'])
    cmip_monthly = xr.open_dataset(snakemake.input['cmip_monthly'])
    msl_monthly_fname = snakemake.output[0]
    exp = snakemake.wildcards['exp']
    # -Script-----------------------------------------------------------
    # Correct zos and zostoga datums
    cmip_monthly_minus_datum = cmip_monthly - datums
    # Get ice melt from MAGICC output
    te, im = get_ice_melt(magicc, exp)
    # Correct ice melt datums and convert cm to m
    te_minus_datum = (te - te[slice("1983-01-01", "2001-12-31")].mean()) * 0.01
    im_minus_datum = (im - im[slice("1983-01-01", "2001-12-31")].mean()) * 0.01
    # Apply TE:IM ratio to zostoga to estimate ice melt
    zostoga = cmip_monthly_minus_datum['zostoga']
    zostoga_plus_im = get_ice_melt_plus_zostoga(
        zostoga, te_minus_datum, im_minus_datum
    )
    # Add local zos to global zostoga + ice melt
    zos = cmip_monthly_minus_datum['zos']
    msl = xr.merge([zostoga_plus_im, zos])
    msl['msl'] = msl['zostoga_plus_ice_melt'] + msl['zos']
    # Add netcdf metadata
    msl.attrs = {
        'ice_melt': "Global monthly ice melt (units: m)",
        'zostoga_plus_ice_melt': (
            "Global monthly average mean sea level from thermosteric"
            "(CMIP zostoga variable) and ice melt. (units: m)"
        ),
        'zos': "Local sterodynamic sea level directly from CMIP (units: m)",
        'zostoga': "Global thermosteric sea level directly from CMIP (units: m)",
        'msl': "Sum of global_msl and zos (units: m)"
    }
    # Write to file
    msl.to_netcdf(msl_monthly_fname)


if __name__ == "__main__":
    main()
