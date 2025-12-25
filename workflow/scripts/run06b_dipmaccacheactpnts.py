
"""Cache actpnts to speed up code."""

import os
import itertools
import pandas as pd
import numpy as np
from snakemake.script import snakemake
import scipy.stats
import scipy.special
from numpy.polynomial.hermite import hermgauss
from aux import stats as s

# Precompute GH nodes/weights
GH_N = 40
t, w = hermgauss(GH_N)
scale = np.sqrt(2.0)
norm_factor = 1.0 / np.pi
X_grid = scale * t[:, None]
Z_grid = scale * t[None, :]
W_grid = w[:, None] * w[None, :]
EPS = 1e-12  # small tolerance for clipping


def integrate_gauss_hermite_batch(rhoz_array, margdist, distarg, p0=0.0, eps=EPS):
    """Vectorized Gauss-Hermite quadrature over rhoz"""
    rhoz_array = np.array(rhoz_array, dtype=float)
    n_rhoz = len(rhoz_array)
    vals = np.empty(n_rhoz, dtype=float)
    # Central moments
    dist = getattr(scipy.stats, margdist)
    m1 = dist.mean(**distarg)
    m2 = dist.var(**distarg)
    for i, rhoz in enumerate(rhoz_array):
        # 2D GH mesh
        Y_grid = rhoz * X_grid + np.sqrt(1.0 - rhoz**2) * Z_grid
        # Standard normal CDFs
        Phi_X = 0.5 * scipy.special.erfc(-X_grid / np.sqrt(2.0))
        Phi_Y = 0.5 * scipy.special.erfc(-Y_grid / np.sqrt(2.0))
        # Transform for marginal, clip to (eps,1-eps)
        denom = 1.0 - p0
        u = np.clip((Phi_X - p0) / denom, eps, 1 - eps)
        v = np.clip((Phi_Y - p0) / denom, eps, 1 - eps)
        # Vectorized quantiles
        A = s.qdist(margdist, u.squeeze(), **distarg)
        B = s.qdist(margdist, v.squeeze(), **distarg)
        # GH quadrature
        vals[i] = norm_factor * np.sum(W_grid * (A * B))
    # Convert to rho_x
    rhox_array = (vals - m1**2) / m2
    return rhox_array


def actpnts_optimized_batch(margdist, margarg, decimal_places, p0=0.0):
    """Actpnts calculation for one marginal"""
    rhoz_values = np.concatenate([np.arange(0.1, 1.0, 0.1), [0.95]])
    rhox_values = integrate_gauss_hermite_batch(
        rhoz_values, margdist, margarg, p0=p0)
    df = pd.DataFrame({
        'rhoz': np.round(rhoz_values, decimal_places),
        'rhox': np.round(rhox_values, decimal_places),
    })
    return df


def actpnts_wrapper(args):
    """Wrapper function for multiprocessing"""
    margdist, marg_tuple, marg_keys, decimal_places = args
    margarg = dict(zip(marg_keys, marg_tuple))
    df = actpnts_optimized_batch(margdist, margarg, decimal_places)
    # add marginal parameters
    for k, v in margarg.items():
        df[k] = v
    df.loc[df.rhox < 0, 'rhox'] = np.nan
    return df.to_dict('records')


def correct_zero_scale(df, min_tol):
    """Correct zero values of the scale parameters to the rounding tolerance.
    """
    df.loc[df['scale'] < min_tol, 'scale'] = min_tol
    return df.drop_duplicates()


def main(param_csvs, output_actpnts_cache_csv, marginal_family,
         precision=0.0001, USE_MP=True):
    decimal_places = int(-np.log10(precision))
    param_names = s.distribution_param_names(marginal_family)
    batch = int(snakemake.params.batch)
    # load parameters
    df = pd.read_parquet(snakemake.input.batched_marginals)
    marg_params = (
        df[df["batch"] == batch]
        .drop(columns="batch")
        .reset_index(drop=True)
        .drop_duplicates()
    )
    marg_params = correct_zero_scale(
        marg_params,
        min_tol=precision,
    )
    args_iter = (
        (marginal_family, tuple(row), param_names, decimal_places)
        for row in marg_params.itertuples(index=False, name=None)
    )
    # compute actpnts
    actpnts_list = [actpnts_wrapper(args) for args in args_iter]
    # flatten and combine into DataFrame
    actpnts_flat = list(itertools.chain.from_iterable(actpnts_list))
    df_actpnts = pd.DataFrame(actpnts_flat)
    df_actpnts.to_csv(output_actpnts_cache_csv, index=False)


def parse_args():
    """Snakemake parser"""
    args = dict(
        param_csvs=snakemake.input,
        output_actpnts_cache_csv=snakemake.output['actpnts'],
        marginal_family=snakemake.wildcards['margfamily'],
        precision=0.0001,
        USE_MP=True
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
