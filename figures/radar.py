# Radar plot of metrics for sample locations and/or overall mean?
# Or show mean by coast?

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd


def radar_plot(
    data_dict,
    categories=None,
    title="Radar Plot",
    ylim=(0, 1),
    colors=None,
    radial_ticks=4,
    # labels placed at this multiple of max radius (data units) -> guarantees a circle
    label_radius=1.15,
    label_style="horizontal",  # "horizontal" (default) or "tangent"
    ax=None,
    shift_angles=False
):
    """
    Create a radar (spider) plot for multiple models or datasets.
    """
    # Validate and (optionally) normalize
    first = next(iter(data_dict.values()))
    num_vars = len(first)
    for v in data_dict.values():
        if len(v) != num_vars:
            raise ValueError("All data series must have the same length.")
    if categories is None:
        categories = [f"Var{i+1}" for i in range(num_vars)]
    if len(categories) != num_vars:
        raise ValueError(
            "Length of categories must match number of variables.")
    # Angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    if shift_angles:
        angles += shift_angles
    angles_closed = np.concatenate([angles, [angles[0]]])
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
    # Plot data
    if colors is None:
        colors = [None] * len(data_dict)
    for (name, values), color in zip(data_dict.items(), colors):
        values = np.array(values, dtype=float)
        closed = np.concatenate([values, [values[0]]])
        ax.plot(angles_closed, closed, linewidth=2, label=name, color=color)
        ax.fill(angles_closed, closed, alpha=0.25, color=color)

    # Radial limits / ticks
    ax.set_ylim(*ylim)
    # ticks = np.linspace(ylim[0], ylim[1], radial_ticks + 1)[1:]  # skip center
    ticks = np.linspace(ylim[0], ylim[1], radial_ticks + 1)
    ax.set_yticks(ticks)
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], alpha=0.6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    # Replace default xticklabels with texts positioned in *data (theta, r)* coordinates.
    # This guarantees labels lie on a perfect circle (radius = label_radius * r_max).
    r_max = ylim[1]
    ax.set_xticks(angles)           # keep ticks for gridlines
    ax.set_xticklabels([])          # hide default tick labels
    r_label = r_max * label_radius
    for theta, txt in zip(angles, categories):
        deg = np.degrees(theta)
        if label_style == "tangent":
            rot = deg if deg <= 180 else deg - 180
            ha = "left" if 0 <= deg <= 180 else "right"
        else:  # horizontal
            rot = 0
            ha = "right" if (90 <= deg <= 270) else "left"
        ax.text(
            theta,
            r_label,
            txt,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            clip_on=False,
        )
    # Title / legend
    ax.set_title(title, fontweight="bold", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.4, -0.1))
    ax.set_aspect("equal", "box")
    return ax


def get_radar(ds):
    return (
        ds
        .to_dataframe()
        .dropna()
        .reset_index(drop=True)
        .mean(axis=0)
    )


def display_names(name):
    name_dict = {
        'nse': "NSE",
        'nse_mod': "Modified\nNSE",
        'pearson_r': r"Pearson $\rho$",
        'rmse': "RMSE",
        'rsq': r"$R^2$",
        'spearman_r':  r"Spearman $\rho$",
        'kge_2009': "KGE\n(2009)",
        'kge_2012': "KGE\n(2012)",
        'mae': "MAE",
        'mse': "MSE",
        'p0': r"$p_0$",
        'loc': "Location\nParam",
        'scale': "Scale Param",
        'cvm_pval': "Cramer\nVon Mises\np-value",
        'ks_pval': "KS Test\np-value",

    }
    return name_dict[name]


def regression_plot_lf(regression_lf, ax):
    regression_lf_radar = pd.DataFrame(
        {"LF":
         get_radar(regression_lf)
         .drop(["record_length", "f_stat_1",
                "nse_rel", "m", "b", "ned", "ed", "stdev", "r2",
                "f_stat_p_val_1", "max_nino34", "min_nino34"])
         })
    ax = radar_plot(
        {'.': regression_lf_radar['LF']},
        [display_names(x) for x in regression_lf_radar.index],
        ylim=(0, 0.2),
        colors=["xkcd:teal"],
        title="Regression (low frequency)",
        ax=ax
    )
    ax.get_legend().remove()
    return ax


def regression_plot_hf(regression_hf, ax):
    regression_hf_radar = get_radar(regression_hf)
    regression_hf_radar = pd.DataFrame(
        {"HF":
         regression_hf_radar
         .drop(
             [
                 x for x in regression_hf_radar.index
                 if ("f_stat" in x) or ("fstat" in x)
                 or ("z_" in x) or ("min_" in x) or ("max_" in x)
             ] +
             [
                 "record_length", "nse_rel", "m_slp",
                 "m_tauu", "m_tauv", "b", "ned", "ed", "r2"
             ]
         )
         }
    )
    ax = radar_plot(
        {
            '.': regression_hf_radar['HF']
        },
        [display_names(x) for x in regression_hf_radar.index],
        ylim=(0, 1),
        colors=["xkcd:teal"],
        title="Regression (high frequency)",
        ax=ax
    )
    ax.get_legend().remove()
    return ax


def acs_plot(dipmac_acs_paretoII, dipmac_acs_weibull, ax):
    dipmac_acs_paretoII_radar = pd.DataFrame(
        {'Pareto II': get_radar(dipmac_acs_paretoII)})
    dipmac_acs_weibull_radar = pd.DataFrame(
        {'Weibull': get_radar(dipmac_acs_weibull)})
    acs = dipmac_acs_paretoII_radar.join(dipmac_acs_weibull_radar)
    acs = acs.drop(["record_length", "nse_rel", "ned", "ed", "scale", "shape"])
    ax = radar_plot(
        {
            'Pareto II': acs['Pareto II'],
            'Weibull': acs['Weibull']},
        [display_names(x) for x in acs.index],
        ylim=(0, 1),  # (acs.min().min(), acs.max().max()),
        colors=["xkcd:teal", "xkcd:coral"],
        title="ACS Fit",
        ax=ax
    )
    return ax


def marg_plot(dipmac_marg_t, dipmac_marg_skewnorm, ax):
    dipmac_marg_t_radar = pd.DataFrame(
        {'t': get_radar(dipmac_marg_t)}).drop('d')
    dipmac_marg_skewnorm = pd.DataFrame(
        {'skewnorm': get_radar(dipmac_marg_skewnorm)}).drop('a')
    marg = dipmac_marg_t_radar.join(dipmac_marg_skewnorm)
    marg = marg.drop('record_length')
    # reorder
    marg = marg.loc[['loc', 'scale', 'ks_pval', 'cvm_pval']]
    ax = radar_plot(
        {
            'Skewnorm': marg['skewnorm'],
            "Student's t": marg['t'],
        },
        [display_names(x) for x in marg.index],
        ylim=(marg.min().min(), marg.max().max()),
        colors=["xkcd:teal", "xkcd:coral"],
        title="Marginal Fit",
        ax=ax,
        shift_angles=np.pi/6
    )
    return ax


def main():
    PARAM_BASE_DIR = Path("/media/annika/blue/cmipsl/06_final/params")
    regression_lf = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_daily_lf_regression.nc")
    )
    regression_hf = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_daily_hf_regression.nc")
    )
    dipmac_acs_paretoII = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_hourly_dipmac_acs_params_paretoII.nc")
    )
    dipmac_acs_weibull = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_hourly_dipmac_acs_params_weibull.nc")
    )
    dipmac_marg_t = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_hourly_dipmac_marg_params_t.nc")
    )
    dipmac_marg_skewnorm = xr.open_dataset(
        PARAM_BASE_DIR.joinpath("ntr_hourly_dipmac_marg_params_skewnorm.nc")
    )
    # Param means for radar plot
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.5), subplot_kw={'projection': 'polar'},
                            constrained_layout=True)  # ideal: figsize=(9, 7)
    for ax in axs.flat:
        ax.set_aspect("equal", "box")
    axs[0, 0] = regression_plot_lf(regression_lf, axs[0, 0])
    axs[0, 1] = regression_plot_hf(regression_hf, axs[0, 1])
    axs[1, 0] = marg_plot(dipmac_marg_t, dipmac_marg_skewnorm, axs[1, 0])
    axs[1, 1] = acs_plot(dipmac_acs_paretoII, dipmac_acs_weibull, axs[1, 1])
    plt.savefig("figures/radar.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
