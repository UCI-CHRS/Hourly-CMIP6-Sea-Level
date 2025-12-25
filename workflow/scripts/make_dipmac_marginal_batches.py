
import os
import numpy as np
import pandas as pd
from snakemake.script import snakemake
from aux import stats as s

precision = 0.0001
decimal_places = int(-np.log10(precision))

param_names = s.distribution_param_names(snakemake.wildcards.margfamily)

param_dfs = [
    pd.read_csv(f, index_col="ts", parse_dates=True)
    for f in snakemake.input.dipmaccache_input
    if os.stat(f).st_size > 0
]

dfs = [
    df[['param', 'hourly_cmip']]
    .pivot(columns="param")['hourly_cmip']
    .dropna()
    .reset_index()[param_names]
    .round(decimal_places)
    .drop_duplicates()
    for df in param_dfs
]

marg_params = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

# Correct zero scale
min_tol = precision
if "scale" in marg_params.columns:
    marg_params.loc[marg_params["scale"] < min_tol, "scale"] = min_tol

# Assign batches
batch_size = snakemake.params.batch_size
marg_params["batch"] = np.arange(len(marg_params)) // batch_size

marg_params.to_parquet(snakemake.output.batched_marginals)
