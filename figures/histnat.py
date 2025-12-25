"""PR, FAR for the hist-nat runs"""

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


def is_pct(array, pct, tol=1e-7):
    """x: array like; pct: float"""
    return [(x - tol < pct) and (x + tol > pct) for x in array]


def corresponding_hist_one_coord(df, percentile):
    """df: dataframe with pct, nat, hist columns"""
    nat_pctl = df.loc[is_pct(df['pct'], percentile)]['nat'].values[0]
    p1 = np.interp(nat_pctl, df['hist'], df['pct'])
    return p1


def attribution_one_id(mid, percentile):
    df_nat = (
        pd.read_csv(
            f"{DATA_DIR_OUT}/{mid}_hist-nat_skewnorm_weibull_twl.csv"
        ).set_index(['x', 'y', 'pct'])
        .rename({'twl': 'nat'}, axis=1)
    )
    df_hist = (
        pd.read_csv(
            f"{DATA_DIR_OUT}/{mid}_historical_skewnorm_weibull_twl.csv"
        ).set_index(['x', 'y', 'pct'])
        .rename({'twl': 'hist'}, axis=1)
    )
    df = df_nat.join(df_hist).reset_index()  # .set_index(['x', 'y'])
    p1 = df.groupby(['x', 'y']).apply(
        lambda x: corresponding_hist_one_coord(x, percentile),
        include_groups=False
    )
    pr = p1/percentile
    far = 1 - percentile/p1
    return {'pr': pr, 'far': far}


def attribution(percentile):
    fnames = list(Path(DATA_DIR_OUT).glob(
        "*_hist-nat_skewnorm_weibull_twl.csv")
    )
    ids = [str(f.stem).split("_hist-nat", maxsplit=1)[0] for f in fnames]
    att = [
        attribution_one_id(mid, percentile) for mid in ids
    ]
    pr_mean = pd.concat([x['pr'] for x in att], axis=1).mean(axis=1)
    far_mean = pd.concat([x['far'] for x in att], axis=1).mean(axis=1)
    return pr_mean, far_mean


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
    percentiles = [0.75, 0.9]
    crs_epsg = ccrs.Mercator()
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 5),
        subplot_kw={"projection": crs_epsg},
        dpi=100
    )
    # Write percentiles if needed
    for sim in ['historical', 'hist-nat']:
        write_percentiles_per_model(sim)
    # Get PR and FAR
    vmin = 0.95
    vmax = 1.05
    cmap = "RdBu_r"
    for ax, percentile in zip(axes.ravel(), percentiles):
        pr, far = attribution(percentile)
        ds = pr.to_xarray()
        cbar_label = f"Probability Ratio ({int(100*percentile)}th percentile)"
        # Map values
        ax_map = draw_map(ax)
        ax_map, _, _ = plot_coastal_pcolormesh(
            ds, ax=ax_map, add_colorbar=True,  # False,
            vmin=vmin, vmax=vmax, cmap=cmap,
            cbar_kwargs={'fraction': 0.046, 'pad': 0.02,
                         'orientation': 'horizontal'},
            cbar_label=cbar_label
        )
        # ax_map.set_title(sim.upper())
        ax_map.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
        gl = ax_map.gridlines(draw_labels=False)
        gl.xlines = gl.ylines = False
    plt.savefig("figures/histnat.png", bbox_inches="tight", dpi=300)


# def main():
#     percentile = 0.9
#     crs_epsg = ccrs.Mercator()
#     fig, axes = plt.subplots(
#         nrows=1, ncols=2,
#         figsize=(10, 5),
#         subplot_kw={"projection": crs_epsg},
#         dpi=100
#     )
#     # Write percentiles if needed
#     for sim in ['historical', 'hist-nat']:
#         write_percentiles_per_model(sim)
#     # Get PR and FAR
#     pr, far = attribution(percentile)
#     ds_list = [pr.to_xarray(), far.to_xarray()]
#     cbar_labels = ['Probability Ratio', 'Fraction of Attributable Risk']
#     vmins = [0.96, -0.04]
#     vmaxs = [1.04, 0.04]
#     cmap = "RdBu_r"
#     for ax, ds, cbar_label, vmin, vmax in zip(axes.ravel(), ds_list, cbar_labels, vmins, vmaxs):
#         # Map values
#         ax_map = draw_map(ax)
#         ax_map, _, _ = plot_coastal_pcolormesh(
#             ds, ax=ax_map, add_colorbar=True,  # False,
#             vmin=vmin, vmax=vmax, cmap=cmap,
#             cbar_kwargs={'fraction': 0.046, 'pad': 0.02,
#                          'orientation': 'horizontal'},
#             cbar_label=cbar_label
#         )
#         # ax_map.set_title(sim.upper())
#         ax_map.tick_params(left=False, bottom=False,
#                            labelleft=False, labelbottom=False)
#         gl = ax_map.gridlines(draw_labels=False)
#         gl.xlines = gl.ylines = False
#     plt.show()
#     breakpoint()
#     # plt.savefig("figures/histnat.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
