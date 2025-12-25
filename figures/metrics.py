"""Matrix of validation metrics"""

import os
# os.chdir(os.path.dirname(os.getcwd()))
# breakpoint()

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
import matplotlib as mpl
import pandas as pd
import xarray as xr
import numpy as np
import scipy
from workflow.aux import stats


def format_matrix_plot(fig, ax, im, cbar_label, df=None, variable=None):
    # Shrink colorbar
    cbar = fig.colorbar(im, ax=ax)
    pos_ax = ax.get_position()
    pos_cbar = cbar.ax.get_position()
    # Update the colorbar to match the y-position and height of ax
    cbar.ax.set_position([
        pos_cbar.x0,      # keep current x-position
        pos_ax.y0,        # align bottom with main ax
        pos_cbar.width,   # keep current width
        pos_ax.height     # match height of main ax
    ])

    # Set x-ticks based on actual time coordinates
    if df is not None and variable is not None:
        time_coords = df[variable].coords['time'].values
        n_times = len(time_coords)
        # Set tick positions at every month (or adjust spacing as needed)
        tick_positions = np.arange(n_times)
        ax.set_xticks(tick_positions)
        # Convert to pandas datetime for formatting
        # time_labels = pd.to_datetime(time_coords)
        # ax.set_xticklabels([t.strftime('%b') for t in time_labels])
        # Or if you want to show fewer ticks to avoid crowding:
        tick_positions = np.linspace(0, n_times-1, 12, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([pd.to_datetime(time_coords[i]).strftime('%b')
                            for i in tick_positions])

    # Add a sideways label
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    return ax, cbar


def make_one_matrix_pvalue(df, variable, fig, ax, cbar_label, pval_thresh=0.05):
    divnorm = colors.TwoSlopeNorm(
        vmin=np.log10(df[variable].min()),
        vcenter=np.log10(0.05),
        vmax=np.log10(df[variable].max())
    )
    im = ax.imshow(df[variable].pipe(np.log10), cmap="RdBu", norm=divnorm,
                   aspect='auto')
    ax, cbar = format_matrix_plot(
        fig, ax, im, cbar_label, df=df, variable=variable)
    cbar.set_ticks([
        np.log10(df[variable].min()),
        np.log10(pval_thresh),
        np.log10(df[variable]).max()
    ])
    cbar.set_ticklabels(["0", f"{pval_thresh}", "1"])
    return ax


def make_one_matrix_R2(df, variable, fig, ax, cbar_label):
    im = ax.imshow(
        df[variable], cmap="Blues", vmin=0, vmax=1,
        aspect='auto')
    ax, cbar = format_matrix_plot(
        fig, ax, im, cbar_label, df=df, variable=variable)
    cbar.set_ticks([0, 0.5, 1])
    return ax


def index_grid_cells(df):
    df_mod = df.copy()
    df_mod['dist_sw'] = np.sqrt(
        (df_mod.y - df_mod.y.min())**2 + (df_mod.x - df_mod.x.min())**2)
    df_mod['ind_geo'] = df_mod.groupby("time")['dist_sw'].transform(
        lambda df: scipy.stats.rankdata(df, method="min"))
    df_mod = df_mod.set_index(['ind_geo', 'time'])
    df_mod = df_mod.to_xarray()
    return df_mod


def open(fname):
    basedir = Path(
        "/media/annika/blue/cmipsl/04_params/dipmac_regression/grd001/src001"
    )
    df = (
        xr.open_dataset(
            basedir.joinpath(fname)
        )
        .to_dataframe()
        .dropna()
        .reset_index()
    )
    # DiPMaC not clipped to conus
    xmin, xmax = -127, -64
    ymin, ymax = 22, 52
    df = df.loc[[
        (xmin <= x <= xmax) &
        (ymin <= y <= ymax)
        for x, y in zip(df.x, df.y)
    ]]
    ds = index_grid_cells(df)
    return ds


def main():
    regression_daily = open("ntr_daily_regression.nc")
    regression_monthly = open("ntr_monthly_regression.nc")
    marg_t_hourly = open(
        "dipmac001/ntr_hourly_dipmac_marg_params_t.nc")
    marg_skewnorm_hourly = open(
        "dipmac001/ntr_hourly_dipmac_marg_params_skewnorm.nc")
    acs_paretoII_hourly = open(
        "dipmac001/ntr_hourly_dipmac_acs_params_paretoII.nc")
    acs_weibull_hourly = open(
        "dipmac001/ntr_hourly_dipmac_acs_params_weibull.nc")
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6, 5))
    fig.subplots_adjust(hspace=0.4)
    mtrx1 = make_one_matrix_pvalue(
        marg_skewnorm_hourly, 'ks_pval', fig, ax[0], "K-S p-val"
    )
    mtrx1.set_title("NTR Marginal Distribution Fit (skewnorm)")
    mtrx2 = make_one_matrix_R2(acs_weibull_hourly, 'rsq', fig, ax[1], "$R^2$")
    mtrx2.set_title("NTR Autocorrelation Structure (weibull)")
    mtrx3 = make_one_matrix_R2(regression_monthly, 'rsq', fig, ax[2], "$R^2$")
    mtrx3.set_title("ENSO Regression (low frequency NTR)")
    mtrx4 = make_one_matrix_R2(regression_daily, 'rsq', fig, ax[3], "$R^2$")
    mtrx4.set_title("Wind + SLP Regression (high frequency NTR)")
    ax[3].set_xlabel("Center of 30 day moving window")
    fig.supylabel('Grid cell index (ordered by distance from southwest)')

    plt.savefig("figures/metrics.png",
                dpi=300, bbox_inches="tight")
    plt.close('all')


if __name__ == "__main__":
    main()
