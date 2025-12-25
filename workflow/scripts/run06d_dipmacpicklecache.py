
"""Pickle the nkblocks and actf caches to speed up dipmacrun.
"""

import pickle
import gc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
import os
try:
    from snakemake.script import snakemake
except ImportError:
    pass
from aux import stats as s


def rss_gb():
    """Check memory."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def build_actf_cache(cache_df, param_cols):
    actf_cols = [c for c in cache_df.columns if 'param' in c]
    cache = {}
    for row in cache_df.itertuples(index=False):
        key = tuple(getattr(row, c) for c in param_cols)
        val = np.array([getattr(row, c) for c in actf_cols], dtype=np.float64)
        cache[key] = val
    return cache


def build_nkblocks_cache_from_parquet(
    parquet_path,
    param_cols,
    max_kblocks,
    batch_rows=500_000,
):
    """
    Stream nkblocks parquet → dict[(params...)] = np.array([[xk, v], ...])
    Vectorized batch-level processing for speed.

    Assumes parquet is sorted by param_cols, then xk.
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(
        columns=param_cols + ["xk", "v", "Fxk"],
        batch_size=batch_rows,
    )
    cache = {}
    min_F = (1.0 - max_kblocks) / 2.0
    max_F = max_kblocks + min_F
    current_key = None
    # Buffers for one group
    buf_xk = []
    buf_v = []
    buf_Fxk = []

    def flush_current():
        """Finalize one group and store in cache."""
        nonlocal buf_xk, buf_v, buf_Fxk, current_key
        if current_key is None:
            return
        # concatenate all batch slices for this group
        xk_arr = np.concatenate(buf_xk)
        v_arr = np.concatenate(buf_v)
        Fxk_arr = np.concatenate(buf_Fxk)
        # clip v values
        mask = (Fxk_arr >= min_F) & (Fxk_arr <= max_F)
        vmax = v_arr[mask].max()
        v_arr = np.minimum(v_arr, vmax)
        cache[current_key] = np.column_stack((xk_arr, v_arr))
        buf_xk.clear()
        buf_v.clear()
        buf_Fxk.clear()
    # Process batch-by-batch
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    processed_rows = 0
    with tqdm(total=total_rows, desc="Processing nkblocks rows") as pbar:
        for batch in scanner.to_batches():
            n = batch.num_rows
            # Convert batch to dict of numpy arrays
            cols = batch.to_pydict()
            # stack param columns into 2D array
            params_arr = np.column_stack([cols[c] for c in param_cols])
            xk_arr = np.asarray(cols["xk"])
            v_arr = np.asarray(cols["v"])
            Fxk_arr = np.asarray(cols["Fxk"])
            # Find where keys change inside the batch
            # 1. Compare adjacent rows
            if len(params_arr) == 0:
                continue
            # Compute boolean array: True where key changes
            key_change = np.any(np.diff(params_arr, axis=0) != 0, axis=1)
            # add True at the last row to ensure last group is flushed
            key_change = np.append(key_change, True)
            start_idx = 0
            # Iterate over group boundaries
            for end_idx in np.where(key_change)[0] + 1:
                key = tuple(params_arr[start_idx])
                # If current group is same as previous, append
                if current_key is None:
                    current_key = key
                if key != current_key:
                    flush_current()
                    current_key = key
                buf_xk.append(xk_arr[start_idx:end_idx])
                buf_v.append(v_arr[start_idx:end_idx])
                buf_Fxk.append(Fxk_arr[start_idx:end_idx])
                start_idx = end_idx
            del batch, cols, params_arr, xk_arr, v_arr, Fxk_arr
            gc.collect()
            processed_rows += n
            pbar.update(n)
        # flush last group
        flush_current()
    return cache


def save_cache(cache, filename):
    with open(filename, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(
    nkblocks_cache_parquet: str,
    actf_cache_parquet: str,
    marginal_family: str,
    max_kblocks: float,
    nkblocks_cache_pkl: str,
    actf_cache_pkl: str,


):
    """Main script to run via snakemake."""
    # Read .parquet files
    param_cols = s.distribution_param_names(marginal_family)
    # Build O(1) nkblocks and actf caches to speed up
    # nkblocks cache
    print("Building nkblocks cache (streaming)...")
    nkblocks_cache = build_nkblocks_cache_from_parquet(
        nkblocks_cache_parquet,
        param_cols,
        max_kblocks,
    )
    save_cache(nkblocks_cache, nkblocks_cache_pkl)
    del nkblocks_cache
    gc.collect()
    # actf cache
    actf_cache_df = pd.read_parquet(actf_cache_parquet)
    print("Building actf cache...")
    actf_cache = build_actf_cache(actf_cache_df, param_cols)
    del actf_cache_df
    gc.collect()
    save_cache(actf_cache, actf_cache_pkl)


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        nkblocks_cache_parquet=snakemake.input['nkblocks_cache_parquet'],
        actf_cache_parquet=snakemake.input['actf_cache_parquet'],
        marginal_family=snakemake.wildcards['margfamily'],
        max_kblocks=snakemake.params['max_kblocks'],
        nkblocks_cache_pkl=snakemake.output['nkblocks_cache_pkl'],
        actf_cache_pkl=snakemake.output['actf_cache_pkl'],
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
