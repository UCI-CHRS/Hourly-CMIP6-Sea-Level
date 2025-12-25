
"""Run DiPMaC for temporal disaggregation of daily NTR."""

import pickle
import os
import datetime as dt
import pandas as pd
import numpy as np
import tqdm
try:
    from snakemake.script import snakemake
except ImportError:
    pass
from aux import stats as s
import aux.dipmac_numba as dipmac


def build_daily_param_cache(param_df: pd.DataFrame, value_colname: str):
    """
    Build fast lookup cache: (month, day) -> dict[param -> value]
    """
    # Extract month/day once (fast NumPy ops)
    md = np.column_stack((param_df.index.month, param_df.index.day))
    # Convert param/value columns to arrays
    params = param_df["param"].to_numpy()
    values = param_df[value_colname].to_numpy()
    # Unique (month, day)
    unique_md, inverse = np.unique(md, axis=0, return_inverse=True)
    # Build cache dict
    cache = {}
    for i, (m, d) in enumerate(unique_md):
        # rows belonging to this month/day
        rows = np.where(inverse == i)[0]
        if len(rows):
            # build param→value mapping
            cache[(m, d)] = dict(zip(params[rows], values[rows]))
        else:
            cache[(m, d)] = {}
    return cache


def get_actf_from_cache_lookup(marg_params_dict, cache, param_cols):
    """
    Fast O(1) lookup in the prebuilt ACTF cache.
    """
    key = tuple(marg_params_dict[k] for k in param_cols)
    return cache.get(key, None)


def nearest_value(arr, xk):
    """
    arr: NumPy array sorted by xk: shape (N, 2) where col0 = xk, col1 = v
    xk: query point

    Returns v corresponding to nearest xk.
    """
    xs = arr[:, 0]
    idx = np.searchsorted(xs, xk)
    if idx == 0:
        return arr[0, 1]
    if idx == len(xs):
        return arr[-1, 1]
    # Compare the neighbor on the left and right
    left = idx - 1
    right = idx
    if abs(xs[left] - xk) <= abs(xs[right] - xk):
        return arr[left, 1]
    else:
        return arr[right, 1]


def get_nkblocks_from_cache_lookup(cache, marg_params_dict, xk, sorted_keys):
    entry = cache.get(tuple(marg_params_dict[k] for k in sorted_keys))
    if entry is None:
        return None
    shard, offset, length = entry
    arr = shard[offset : offset + length]  # zero-copy slice
    return nearest_value(arr, xk)


def get_daily_param_from_cache_lookup(ts, cache):
    return cache.get((ts.month, ts.day), {})


def run_dipmac_moving_window(
    ntr_df: pd.DataFrame,
    marg_cache,
    acs_cache,
    actf_cache,
    nkblocks_cache,
    marginal_family: str,
    acs_family: str,
    lagmax: int,
    csv_out: str
) -> tuple[np.ndarray, np.ndarray]:
    hours_per_day = 24
    n_hours = ntr_df.shape[0] * hours_per_day
    # # Pre-compute marg and acs values
    marg_param_names = s.distribution_param_names(marginal_family)
    acs_param_names = s.distribution_param_names_acf(acs_family)
    sorted_marg_keys = tuple(sorted(marg_param_names))
    marg_all = {
        ts: {k: v for k, v in get_daily_param_from_cache_lookup(
            ts, marg_cache).items() if k in marg_param_names}
        for ts in ntr_df.index
    }
    acs_all = {
        ts: {k: v for k, v in get_daily_param_from_cache_lookup(
            ts, acs_cache).items() if k in acs_param_names}
        for ts in ntr_df.index
    }
    # Use numpy memory map to avoid OOM errors with large concatenated array
    x_values = np.memmap(csv_out, dtype="float32", mode="w+", shape=(n_hours,))
    z_values = np.array([])
    pos = 0
    for ts in tqdm.tqdm(ntr_df.index):
        xk = ntr_df.loc[ts]
        marg = get_daily_param_from_cache_lookup(ts, marg_cache)
        acs = get_daily_param_from_cache_lookup(ts, acs_cache)
        marg = marg_all[ts]
        acs = acs_all[ts]
        acf_val = np.array([s.acf(acs_family, lag, **acs)
                            for lag in range(1, int(lagmax)+1)])
        actfpara = get_actf_from_cache_lookup(
            marg, actf_cache, marg_param_names)
        if actfpara is None:
            x_values[pos:pos +
                     hours_per_day] = np.full(hours_per_day, np.nan)
            z_values = np.full(lagmax, np.nan)
        else:
            n_k_blocks = get_nkblocks_from_cache_lookup(
                nkblocks_cache, marg, xk, sorted_marg_keys
            )
            if len(z_values) > lagmax + 1:
                z_values = z_values[-lagmax:]
            best_kblock = dipmac.iterate_k_blocks(
                xk, int(n_k_blocks), marginal_family, marg, acf_val, actfpara,
                hours_per_day, z_values
            )
            if best_kblock is not None:
                z_values = best_kblock['z']
                x_values[pos:pos+hours_per_day] = best_kblock['x']
            else:
                x_values[pos:pos +
                         hours_per_day] = np.full(hours_per_day, np.nan)
                z_values = np.full(lagmax, np.nan)
        pos += hours_per_day
    x_values.flush()
    start_date = ntr_df.index.min()
    hour_index = np.arange(len(x_values))
    hourly_times = start_date + hour_index.astype("timedelta64[h]")
    ntr_hourly = pd.DataFrame({'ntr': x_values}, index=hourly_times)
    return ntr_hourly


def fix_dates(
    df: pd.DataFrame,
    startyear: int,
    scenario: str
) -> pd.DataFrame:
    """Use equal periods for future and historical simulations, 
    capped by the startyear used to fit params.
    """
    df.index -= dt.timedelta(hours=12)
    range_years = 2015 - startyear
    # historical and hist-nat
    if "hist" in scenario.lower():
        df = df[f"{startyear}-01-01":]
    elif "ssp" in scenario.lower():
        df = df[:f"{2015+range_years}-01-01"]
    return df


def load_cache(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_nkblocks_cache_mmap(outdir: str):
    """
    Load nkblocks cache written by table2npy.py.

    Returns:
        cache: dict
            key -> (mmap_array, offset, length)
    """
    outdir = os.path.abspath(outdir)
    index = np.load(
        os.path.join(outdir, "index.npy"),
        allow_pickle=True
    ).item()
    # Open all shards once
    shards = {}
    for shard_id, _, _ in index.values():
        if shard_id not in shards:
            fname = os.path.join(outdir, f"nkblocks_{shard_id:05d}.npy")
            shards[shard_id] = np.load(fname, mmap_mode="r")
    # Map keys → slice metadata
    cache = {
        key: (shards[shard_id], offset, length)
        for key, (shard_id, offset, length) in index.items()
    }
    return cache


def main(
    lagmax: int,
    startyear: int,
    scenario: str,
    marginal_family: str,
    acs_family: str,
    dipmac_marg_params: str,
    dipmac_actf_cache: str,
    dipmac_nkblocks_cache_npy: str,
    dipmac_acs_params: str,
    daily_ntr_cmip: str,
    hourly_ntr_cmip_out: str,
):
    """Main script to run via snakemake."""
    # read params
    actf_cache = load_cache(dipmac_actf_cache)
    # nkblocks_cache = load_cache(dipmac_nkblocks_cache)
    nkblocks_cache = load_nkblocks_cache_mmap(dipmac_nkblocks_cache_npy)
    marg_df = pd.read_csv(
        dipmac_marg_params,
        index_col='ts',
        parse_dates=True
    )
    acs_df = (
        pd.read_csv(
            dipmac_acs_params,
            index_col=0, parse_dates=['ts']
        ).set_index('ts')
    )
    # Daily param lookup cache
    marg_cache = build_daily_param_cache(marg_df, "hourly_cmip")
    acs_cache = build_daily_param_cache(acs_df, "value")
    daily_ntr_cmip_df = pd.read_csv(
        daily_ntr_cmip, index_col='time', parse_dates=True
    )
    daily_ntr_cmip_df = fix_dates(
        daily_ntr_cmip_df['ntr_total'],
        startyear,
        scenario
    )
    print("Running main loop...")
    ntr_hourly = run_dipmac_moving_window(
        ntr_df=daily_ntr_cmip_df,
        marg_cache=marg_cache,
        acs_cache=acs_cache,
        actf_cache=actf_cache,
        nkblocks_cache=nkblocks_cache,
        marginal_family=marginal_family,
        acs_family=acs_family,
        lagmax=lagmax,
        csv_out=hourly_ntr_cmip_out
    )
    ntr_hourly.to_csv(hourly_ntr_cmip_out)


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        lagmax=snakemake.params['lagmax'],
        startyear=snakemake.params['startyear'],
        scenario=snakemake.wildcards['exp'],
        marginal_family=snakemake.wildcards['margfamily'],
        acs_family=snakemake.wildcards['acsfamily'],
        dipmac_marg_params=snakemake.input['marg_params'],
        dipmac_actf_cache=snakemake.input['dipmac_actf_cache'],
        dipmac_nkblocks_cache_npy=snakemake.params['dipmac_nkblocks_cache_npy'],
        dipmac_acs_params=snakemake.input['acs_params_hourly_obs'],
        daily_ntr_cmip=snakemake.input['daily_ntr_cmip'],
        hourly_ntr_cmip_out=snakemake.output['hourly_ntr'],
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
