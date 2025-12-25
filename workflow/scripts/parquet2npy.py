#!/usr/bin/env python3
"""
Stream nkblocks CSV -> sharded .npy + index.npy

Usage:
    python csv2npy.py infile.csv outdir \
        --param-cols mu sigma nu \
        --max-kblocks 50
"""

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl
import scipy
from tqdm import tqdm


def param_names(dist: str):
    distobj = getattr(scipy.stats, dist)
    shape_names = []
    if distobj.shapes:
        shape_names = [s.strip() for s in distobj.shapes.split(',')]
    return shape_names + ['loc', 'scale']


def scan_table(path: str):
    """
    Return a Polars LazyFrame for CSV or Parquet input.
    """
    path = Path(path)
    if path.suffix == ".csv":
        return pl.scan_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}")


def table_to_npy(
    infile: str,
    outdir: str,
    param_cols: list[str],
    max_kblocks: int,
    batch_rows: int = 100_000,
    shard_rows: int = 5_000_000,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    min_F = (1.0 - max_kblocks) / 2.0
    max_F = max_kblocks + min_F

    lf = (
        scan_table(infile)
        .select(param_cols + ["xk", "v", "Fxk"])
    )

    index = {}

    shard_id = 0
    shard_data = []
    shard_size = 0

    current_key = None
    xk_buf = []
    v_buf = []
    Fxk_buf = []

    def flush_group():
        nonlocal shard_size

        if not xk_buf:
            return

        Fxk_arr = np.asarray(Fxk_buf)
        v_arr = np.asarray(v_buf)

        mask = (Fxk_arr >= min_F) & (Fxk_arr <= max_F)
        vmax = v_arr[mask].max()
        vv = np.minimum(v_arr, vmax)

        arr = np.column_stack((xk_buf, vv))

        offset = shard_size
        shard_data.append(arr)
        shard_size += arr.shape[0]

        index[current_key] = (shard_id, offset, arr.shape[0])

    def flush_shard():
        nonlocal shard_id, shard_size

        if not shard_data:
            return

        shard = np.concatenate(shard_data, axis=0)
        fname = outdir / f"nkblocks_{shard_id:05d}.npy"
        np.save(fname, shard)

        shard_data.clear()
        shard_size = 0
        shard_id += 1

    # Optional progress bar
    try:
        total_rows = lf.select(pl.count()).collect().item()
        pbar = tqdm(total=total_rows, desc="Streaming input")
    except Exception:
        pbar = tqdm(desc="Streaming input")

    for batch in lf.collect(streaming=True).iter_slices(batch_rows):
        for row in batch.iter_rows(named=True):
            key = tuple(row[c] for c in param_cols)

            if current_key is None:
                current_key = key

            if key != current_key:
                flush_group()

                if shard_size >= shard_rows:
                    flush_shard()

                xk_buf.clear()
                v_buf.clear()
                Fxk_buf.clear()
                current_key = key

            xk_buf.append(row["xk"])
            v_buf.append(row["v"])
            Fxk_buf.append(row["Fxk"])

        pbar.update(len(batch))

    pbar.close()

    flush_group()
    flush_shard()

    np.save(outdir / "index.npy", index)

    print("\nDone.")
    print(f"  Input: {infile}")
    print(f"  Shards written: {shard_id}")
    print(f"  Index entries: {len(index)}")
    print(f"  Output dir: {outdir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Stream nkblocks CSV -> .npy shards")
    p.add_argument("csv", help="Input CSV file")
    p.add_argument("outdir", help="Output directory for .npy shards")
    p.add_argument(
        "--distribution",
        type=str,
        required=True,
        help="Marginal distribution from which to determine param cols",
    )
    p.add_argument(
        "--max-kblocks",
        type=float,
        required=True,
        help="Maximum kblocks percentile used for Fxk clipping",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=100_000,
        help="CSV streaming batch size",
    )
    p.add_argument(
        "--shard-rows",
        type=int,
        default=5_000_000,
        help="Rows per .npy shard",
    )
    return p.parse_args()


def main():
    args = parse_args()
    param_cols = param_names(args.distribution)
    table_to_npy(
        infile=args.csv,
        outdir=args.outdir,
        param_cols=param_cols,
        max_kblocks=args.max_kblocks,
        batch_rows=args.batch_rows,
        shard_rows=args.shard_rows,
    )


if __name__ == "__main__":
    main()
