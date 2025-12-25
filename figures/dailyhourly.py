
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from workflow.aux import stats


def show_point(df1_x, df1_y, df2_x, df2_y, ax, thresh_y=0.95):
    k1 = np.interp(thresh_y, df1_y, df1_x)
    k2 = np.interp(thresh_y, df2_y, df2_x)
    # lines
    ax.vlines(x=k1, ymin=0, ymax=thresh_y, linewidth=0.5, color="r")
    ax.vlines(x=k2, ymin=0, ymax=thresh_y, linewidth=0.5, color="k")
    ax.hlines(y=thresh_y, xmin=min(df1_x.min(), df2_x.min()),
              xmax=k1, linewidth=0.5, color="r")
    ax.hlines(y=thresh_y, xmin=min(df1_x.min(), df2_x.min()),
              xmax=k2, linewidth=0.5, color="k")
    # distribution
    ax.plot(df1_x, df1_y, linestyle='solid', c='red', lw=2, label='Hourly')
    ax.plot(df2_x, df2_y, linestyle='solid', c='black', lw=2, label='Daily')
    # marker
    ax.plot(k1, thresh_y, "ro", markersize=8, mfc='white')
    ax.plot(k2, thresh_y, "ko", markersize=8, mfc='white')
    return ax


def station_geometry(stations_df, crs_epsg):
    x = stations_df['lng']
    y = stations_df['lat']
    # Map
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:4326")
    gdf = gdf.loc[
        (gdf.geometry.x > -130) &
        (gdf.geometry.x < -50) &
        (gdf.geometry.y > 25) &
        (gdf.geometry.y < 50)
    ]
    df_epsg = gdf.to_crs(crs_epsg)
    return df_epsg


def map_stations(stations, axs, crs_epsg):
    df_epsg = station_geometry(stations, crs_epsg)
    # Make the CartoPy plot
    axs.add_geometries(
        df_epsg["geometry"],
        crs=crs_epsg,
    )
    axs.add_feature(
        cfeature.STATES.with_scale("50m"),
        edgecolor="gray",       # outline color
        facecolor="none",      # no fill
        linewidth=1.0
    )
    axs.coastlines()
    # Make the GeoPandas plot
    df_epsg.plot(
        ax=axs,
        markersize=7,
        facecolor="royalblue",
        edgecolor=None
    )
    return axs


def plot_cdf(station_id, ax):
    twl_files = [
        f for f in Path(
            "/media/annika/Annika/cmip6-hourly-sea-level-data/rawdata/obs/ocean/noaa"
        ).glob(f"twl_stationid_{station_id}*.csv")
        if int(str(f)[-8:-4]) <= 2020
    ]
    df_hourly = pd.concat([
        df['value']
        for f in twl_files
        if 'value' in (df := pd.read_csv(f, index_col=0, parse_dates=[0]))
    ])
    df_daily = df_hourly.resample('D').mean()
    data = [df_hourly.dropna(), df_daily.dropna()]
    p1 = stats.ecdf(df_hourly.dropna())
    p2 = stats.ecdf(df_daily.dropna())
    ax = show_point(p1['value'], p1['p'], p2['value'],
                    p2['p'], ax, thresh_y=0.95)
    ax.set_xlabel("Total Water Level (m)")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([0, 1])
    return ax


def station_callout(
    ax_map,
    stations,
    crs_epsg,
    x,
    y,
    text_coords,
    align,
    label
):
    # add circles around both stations
    df = stations.loc[
        [(x_ == x) and (y_ == y) for x_, y_ in zip(stations.lng, stations.lat)]
    ]
    df_xy = station_geometry(df, crs_epsg)
    ax_map = df_xy.plot(
        ax=ax_map,
        color='none',            # no fill color
        edgecolor='royalblue',       # outline color of circles
        marker='o',              # circle marker
        markersize=300,          # adjust size as needed
        linewidth=1.5,            # line width of the circle edge
        zorder=-1
    )
    # Add annotation
    ax_map.annotate(
        label,
        text_coords,
        fontsize=11,
        horizontalalignment=align,
        xycoords=crs_epsg
    )
    return ax_map


def main():
    stations = pd.read_csv(
        "/media/annika/Annika/cmip6-hourly-sea-level-data/metadata/noaa_station_metadata.csv"
    )
    # figsize = (1.3, 3)  # Inches => 1*100 x 3*100 = 100 x 300 pixels
    figsize = (3.5*3, 3)
    fig = plt.figure(figsize=figsize, dpi=100)
    crs_epsg = ccrs.Mercator()
    ax_map = fig.add_subplot(1, 3, 2, projection=crs_epsg)
    ax_map = map_stations(stations, ax_map, crs_epsg)
    station_id_east = 8518750
    station_id_west = 9414290
    x_east = stations.loc[stations.id == station_id_east, "lng"].values[0]
    y_east = stations.loc[stations.id == station_id_east, "lat"].values[0]
    x_west = stations.loc[stations.id == station_id_west, "lng"].values[0]
    y_west = stations.loc[stations.id == station_id_west, "lat"].values[0]
    # Label east
    ax_map = station_callout(
        ax_map, stations, crs_epsg, x_east, y_east,
        text_coords=[ax_map.get_xlim()[1], sum(ax_map.get_ylim())/8],
        align="right",
        label=f"New York City, NY\nStation ID: {station_id_east}")
    # Label west
    ax_map = station_callout(
        ax_map, stations, crs_epsg, x_west, y_west,
        text_coords=[ax_map.get_xlim()[0], 6*sum(ax_map.get_ylim())/8],
        align="left",
        label=f"San Francisco, CA\nStation ID: {station_id_west}")
    ###
    ###
    # Remove all Cartopy axes decorations
    ax_map.set_frame_on(False)
    ax_map.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
    # make axes background match figure
    ax_map.set_facecolor(fig.get_facecolor())
    # Remove gridlines completely
    ax_map.gridlines(draw_labels=False, alpha=0)
    # Remove gridlines if any
    gl = ax_map.gridlines(draw_labels=False)
    gl.xlines = False
    gl.ylines = False
    # Add CDF plots
    ax_west = fig.add_subplot(1, 3, 1)
    ax_west = plot_cdf(station_id_west, ax_west)
    ax_west.set_ylabel("Cumulative Distribution Function")
    ax_west.legend(bbox_to_anchor=(1.5, 0.22))
    ax_east = fig.add_subplot(1, 3, 3)
    ax_east = plot_cdf(station_id_east, ax_east)
    ax_east.set_yticks([])
    ax_east.set_yticklabels([])
    plt.savefig("figures/Figure1.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
