"""Calculate the daily modeled NTR distributions for each moving window.
1. Read CMIP daily NTR files
2. Looping over the param time steps + lat/lons:
    2.1 Fit the daily distribution of CMIP
3. Save to .nc file
"""

from pathlib import Path
import numpy as np
import datetime as dt
import pandas as pd
import tqdm
from snakemake.script import snakemake
from aux import utils, mwm, stats as s


def fit_marg(
    ts: dt.datetime,
    ntr: pd.Series,
    years: range,
    n_days: int,
    marginal_family: str
) -> None:
    """Fit marginal distribution for one spatiotemporal window.

    Args:
        ts (dt.datetime):
        ntr pd.Series:
            Dataset with each variable clipped to the spatiotemporal window
        years (range[int]): for MWM, range of years to use
        n_days (int): # for MWM, days in moving window
        marginal_family (str): marginal family name

    Returns:
        None (fits saved directly to disk)
    """
    # Fit dipmac to values in window
    ntr_window = mwm.make_temporal_moving_window(
        ntr, years, ts.month, ts.day, ts.hour, n_days
    ).dropna()
    if not ntr_window.empty:
        params = s.fit_distribution(
            marginal_family,
            ntr_window.values
        )
        # Save results to dataframe with datetime index
        if params is not None:
            return pd.DataFrame(
                [{'ts': ts, 'param': k, 'value': v} for k, v in params.items()]
            )


def main(
    csv_in: str,
    csv_out: str,
    margfamily: str,
    ndays: int,
    sy: int,
    ey: int
):
    """Main script to run via snakemake."""
    daily_ntr = pd.read_csv(csv_in, index_col=0, parse_dates=True)['ntr_total']
    if not daily_ntr.empty:
        years = range(sy, ey+1)
        # -Script-----------------------------------------------------------
        interval = dt.timedelta(days=1)
        params = [
            fit_marg(ts, daily_ntr, years, ndays, margfamily)
            for ts in tqdm.tqdm(utils.time_steps(interval))
        ]
        # Combine dataframes and write to csv
        marg = [x for x in params if x is not None]
        if len(marg) > 0:
            marg_df = pd.concat(marg)
            marg_df.to_csv(csv_out)
    else:
        Path(csv_out).touch()


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        csv_in=snakemake.input[0],
        csv_out=snakemake.output[0],
        margfamily=snakemake.wildcards['marg'],
        ndays=snakemake.params['ndays'],
        sy=snakemake.params['sy'],
        ey=snakemake.params['ey']
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
