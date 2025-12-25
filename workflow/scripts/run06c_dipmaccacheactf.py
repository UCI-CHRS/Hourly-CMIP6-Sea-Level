
import multiprocessing
import pandas as pd
import numpy as np
import tqdm
from snakemake.script import snakemake
from aux import dipmac
import aux.stats as s


def actpara(df):
    rhox, rhoz = df['rhox'], df['rhoz']
    params = df.drop(['rhox', 'rhoz'], axis=1).drop_duplicates()
    try:
        if (any(rhox < 0) or any(np.isnan(rhox)) or any(np.isinf(rhox))):
            return None
        actf = dipmac.fitactf(rhox, rhoz)
        ind = 1
        for p in actf:
            params[f"param_{ind}"] = np.round(p, 4)
            ind += 1
        return params
    except ValueError:
        return None


def parse_args():
    """Snakemake params, inputs, outputs"""
    args = dict(
        actpnts_df=snakemake.input[0],
        margdist=snakemake.wildcards['margfamily'],
        output_actf_cache_csv=snakemake.output[0]
    )
    return args


def main(
    actpnts_df: str,
    margdist: str,
    output_actf_cache_csv: str
):
    """Main script to run via snakemake."""
    df = pd.read_csv(actpnts_df)
    param_names = s.distribution_param_names(margdist)
    actfpara_list = [actpara(g) for _, g in df.groupby(param_names)]
    df_actpara = pd.concat([x for x in actfpara_list if x is not None])
    df_actpara.to_csv(output_actf_cache_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
