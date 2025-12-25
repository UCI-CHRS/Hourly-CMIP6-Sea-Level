
"""TWL percent change for SSP vs historical - hourly and daily"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr
import numpy as np
import cartopy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from workflow.aux import stats

# -----------------------------------------------------------------------
# TWL NCs
DATA_DIR_IN = "/media/annika/blue/cmipsl/06_final/data"
# TWL percentile CSVs
DATA_DIR_OUT = "/media/annika/blue/cmipsl/06_final/figures"
# -----------------------------------------------------------------------

# Pooled version
# def get_model_average(sim, DATA_DIR_IN, DATA_DIR_OUT):
#     # Get model average by simulation
#     fname_out = Path(DATA_DIR_OUT).joinpath(f"{sim}_percentiles.csv")
#     if not fname_out.exists():
#         ds_list = [
#             xr.open_dataset(f)
#             for f in Path(DATA_DIR_IN).glob(f"*_{sim}_skewnorm_weibull_twl.nc")
#         ]
#         df_list = [ds.to_dataframe().dropna() for ds in ds_list]
#         df = pd.concat([df.reset_index().drop("time", axis=1) for df in df_list])
#         percents = np.arange(0.01, 0.99, 0.01).tolist() + [.999, 0.9999, 0.99999, 1]
#         quantiles = df.groupby(['x', 'y']).quantile(percents)
#         quantiles.to_csv(str(fname_out))
#     return fname_out


def write_percentiles_per_model(sim):
    fnames_nc = list(Path(DATA_DIR_IN).glob(
        f"*_{sim}_skewnorm_weibull_twl.nc"))
    fnames_csv = [
        Path(DATA_DIR_OUT).joinpath(str(f.name).replace(".nc", ".csv"))
        for f in fnames_nc
    ]
    for fname_in, fname_out in zip(fnames_nc, fnames_csv):
        if not fname_out.exists():
            ds = xr.open_dataset(fname_in)
            df = ds.to_dataframe().dropna().reset_index().drop("time", axis=1)
            percents = np.arange(0.01, 0.99, 0.01).tolist() + \
                [.999, 0.9999, 0.99999, 1]
            quantiles = df.groupby(['x', 'y']).quantile(percents).reset_index()
            quantiles = quantiles.rename({quantiles.columns[2]: "pct"}, axis=1)
            quantiles.to_csv(str(fname_out), index=False)


def draw_map(axs):
    axs.add_feature(
        cfeature.STATES.with_scale("50m"),
        edgecolor="gray",       # outline color
        facecolor="none",      # no fill
        linewidth=1.0
    )
    axs.coastlines()
    return axs


def format_colorbar(fig, ax, im, cbar_label):
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
    # Add a sideways label
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    return ax, cbar


def is_pct(array, pct, tol=1e-7):
    """x: array like; pct: float"""
    return [(x - tol < pct) and (x + tol > pct) for x in array]


def percent_change_one_id(mid, sim, percentile):
    df_sim = (
        pd.read_csv(
            f"{DATA_DIR_OUT}/{mid}_{sim}_skewnorm_weibull_twl.csv"
        ).set_index(['x', 'y', 'pct'])
        .rename({'twl': 'sim'}, axis=1)
    )
    df_hist = (
        pd.read_csv(
            f"{DATA_DIR_OUT}/{mid}_historical_skewnorm_weibull_twl.csv"
        ).set_index(['x', 'y', 'pct'])
        .rename({'twl': 'hist'}, axis=1)
    )
    df = df_sim.join(df_hist).reset_index()
    df = (
        df.loc[is_pct(df['pct'], percentile)]
        .drop('pct', axis=1)
        .set_index(['x', 'y'])
    )
    percent_change = 100*(df['sim'] - df['hist'])/df['hist']
    return percent_change


def get_percent_change(sim, percentile):
    fnames_sim = list(Path(DATA_DIR_OUT).glob(
        f"*_{sim}_skewnorm_weibull_twl.csv"))
    ids = [str(f.stem).split(f"_{sim}", maxsplit=1)[0] for f in fnames_sim]
    percent_change = [
        percent_change_one_id(mid, sim, percentile) for mid in ids
    ]
    return percent_change


def plot_coastal_pcolormesh(
    da, ax=None, lon_name="x", lat_name="y",
    cmap="viridis", gap_threshold=5.0,
    add_colorbar=True, cbar_label=None, cbar_kwargs=None, **kwargs
):
    """
    Plot a DataArray with coastal values and large longitude gaps
    using multiple pcolormesh calls to avoid stretched cells.
    
    Parameters
    ----------
    da : xarray.DataArray
        2D array with dimensions (lat, lon).
    ax : matplotlib Axes with Cartopy projection, optional
        If None, a new axis with PlateCarree will be created.
    lon_name, lat_name : str
        Names of longitude and latitude coords.
    cmap : str
        Colormap for plotting.
    gap_threshold : float
        Minimum longitude jump (degrees) that indicates a "gap".
    add_colorbar : bool
        If True, add a colorbar to the axis.
    cbar_kwargs : dict
        Extra keyword args passed to plt.colorbar.
    kwargs : dict
        Extra args passed to ax.pcolormesh.
    
    Returns
    -------
    ax : matplotlib Axes
    cbar : matplotlib Colorbar or None
    """
    lons = da[lon_name].values
    lats = da[lat_name].values
    data = da.transpose(lat_name, lon_name).values  # ensure (lat, lon)
    # find big jumps in longitude
    jumps = np.where(np.diff(lons) > gap_threshold)[0]
    blocks = np.split(lons, jumps + 1)
    data_blocks = np.split(data, jumps + 1, axis=1)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()},
                               figsize=(8, 5))
    mesh = None
    for lon_block, data_block in zip(blocks, data_blocks):
        # compute edges
        def get_edges(coords):
            step = np.diff(coords) / 2
            edges = np.empty(len(coords) + 1)
            edges[1:-1] = coords[:-1] + step
            edges[0] = coords[0] - step[0]
            edges[-1] = coords[-1] + step[-1]
            return edges
        lon_edges = get_edges(lon_block)
        lat_edges = get_edges(lats)
        lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)
        mesh = ax.pcolormesh(
            lon2d, lat2d, data_block,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="flat",
            **kwargs
        )
    ax.coastlines()
    # add colorbar if requested
    cbar = None
    if add_colorbar and mesh is not None:
        if cbar_kwargs is None:
            cbar_kwargs = {}
        cbar = plt.colorbar(mesh, ax=ax, **cbar_kwargs)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    return ax, cbar, mesh


def main():
    sims = ['ssp245', 'ssp585']
    # ['ssp245', 'ssp370', 'ssp434', 'ssp585']  # 'ssp126', 'historical']
    percentile = 0.9
    crs_epsg = ccrs.Mercator()
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 5),
        subplot_kw={"projection": crs_epsg},
        dpi=100
    )
    ds_baseline = None
    for sim, ax in zip(sims, axes.ravel()):
        write_percentiles_per_model(sim)
        # Get percent change
        dfs = get_percent_change(sim, percentile)
        if ds_baseline is None:
            ds = pd.concat(dfs, axis=1).mean(axis=1).to_xarray()
            ds_baseline = ds
            cmap = "Blues"
            vmin, vmax = 0, 150
            cbar_label = f"% Change in {int(percentile*100)}th Percentile from Historical"
        else:
            ds1 = pd.concat(dfs, axis=1).mean(axis=1).to_xarray()
            ds = ds1 - ds
            cmap = "Reds"
            vmin, vmax = 0, 30
            cbar_label = f"Difference in percent change relative to SSP245"
            
        # Map values
        ax_map = draw_map(ax)
        ax_map, cbar, im = plot_coastal_pcolormesh(
            ds, ax=ax_map, add_colorbar=True,  # False,
            vmin=vmin, vmax=vmax, cmap=cmap,
            cbar_kwargs={'fraction': 0.046, 'pad': 0.02, 'orientation': 'horizontal'},
            cbar_label=cbar_label
        )
        ax_map.set_title(sim.upper())
        ax_map.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
        gl = ax_map.gridlines(draw_labels=False)
        gl.xlines = gl.ylines = False
    plt.savefig("figures/ssp.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
