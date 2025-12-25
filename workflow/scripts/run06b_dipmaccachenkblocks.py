
"""Cache the required number of k-blocks for each set of marginal
parameters to speed up code.
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import os
# Set threading env vars explicitly
n_threads = str(snakemake.resources.cpus)   # noqa
os.environ["NUMBA_NUM_THREADS"] = n_threads   # noqa
os.environ["OMP_NUM_THREADS"] = n_threads   # noqa
os.environ["MKL_NUM_THREADS"] = "1"
# import multiprocessing
try:
    from snakemake.script import snakemake
except ImportError:  # won't import during debugging
    pass
from aux.dipmac import number_of_k_blocks, compute_v_numba   # noqa
from aux import stats as s   # noqa
# from aux.dipmac import nkblocks, compute_v_numba


def loop(
    marginal_family,
    marg_params,
    p_e,
    P,
    outfile,
    decimal_places,
):
    """
    Process exactly ONE Snakemake-defined batch.

    Assumptions:
    - marg_params contains only the rows for this batch
    - One process, no multiprocessing
    - Numba handles all parallelism
    - Writes exactly one output file
    """
    # ---- Warm up numba JIT once ----
    dummy = np.linspace(0.0, 1.0, 10)
    _ = compute_v_numba(dummy, dummy, 0.05, 0.95)
    # Stable column order
    param_cols = list(marg_params.columns)
    marg_records = marg_params.to_dict("records")
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xk", "Fxk", "v", *param_cols])
        for marg in tqdm(
            marg_records,
            desc="Running nkblocks (single batch)",
        ):
            xk, Fxk, v = number_of_k_blocks(
                marginal_family=marginal_family,
                marginal_params=marg,
                p_e=p_e,
                P=P,
            )
            marg_values = [marg[k] for k in param_cols]
            for xk_i, Fxk_i, v_i in zip(xk, Fxk, v):
                writer.writerow([
                    round(xk_i, decimal_places),
                    round(Fxk_i, decimal_places),
                    v_i,
                    *marg_values,
                ])


# def loop(
#     marginal_family,
#     marg_params,
#     p_e,
#     P,
#     outfile,
#     decimal_places
# ):
#     """Get nkblocks list with either a list comprehension or multiprocessing."""
#     # Warm up numba JIT once in the parent so workers inherit compiled code
#     _dummy = np.linspace(0, 1, 10)
#     _compute_dummy = compute_v_numba(_dummy, _dummy, 0.05, 0.95)
#     # Choose the order of the parameter columns (stable)
#     param_cols = list(marg_params.columns)
#     with open(outfile, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["xk", "Fxk", "v", *param_cols])
#         args_iter = (
#             (marginal_family, marg, p_e, P)
#             for marg in marg_params.to_dict("records")
#         )
#         n_workers = np.floor(multiprocessing.cpu_count()/2).astype(int)
#         print(f"Number of workers: {n_workers}")
#         with multiprocessing.Pool(n_workers) as pool:
#             for xk, Fxk, v, marg in tqdm(
#                 pool.imap_unordered(nkblocks, args_iter),
#                 total=len(marg_params),
#                 desc="Running nkblocks",
#             ):
#                 marg_values = [marg[k] for k in param_cols]
#                 for xk_i, Fxk_i, v_i in zip(xk, Fxk, v):
#                     writer.writerow([
#                         np.round(xk_i, decimal_places),
#                         np.round(Fxk_i, decimal_places),
#                         v_i,
#                         *marg_values
#                     ])


def correct_zero_scale(df, min_tol):
    """Correct zero values of the scale parameters
    to the rounding tolerance.
    """
    df.loc[df['scale'] < min_tol, 'scale'] = min_tol
    return df.drop_duplicates()


def main(
    p_e: float,
    P: float,
    marginal_family: str,
    dipmac_marg_params: str,
    dipmac_nkblocks_cache: str,
    precision: float = 0.0001
):
    """Main script to run via snakemake."""
    decimal_places = int(-np.log10(precision))
    param_names = s.distribution_param_names(marginal_family)
    batch = int(snakemake.params.batch)
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
    # Create the nkblock cache
    loop(
        marginal_family,
        marg_params,
        p_e,
        P,
        dipmac_nkblocks_cache,
        decimal_places
    )


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        p_e=snakemake.params['p_e'],
        P=snakemake.params['P'],
        marginal_family=snakemake.wildcards['margfamily'],
        dipmac_marg_params=snakemake.input,
        dipmac_nkblocks_cache=snakemake.output['nkblocks'],
        precision=0.0001,
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**args)
