
rule dipmaccmipparams:
    """Calculate dipmac marginal parameter shifts for each CMIP model. 
    """
    input:
        marg_params_hourly_obs = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "lat_{lat}_lon_{lon}_ntr_hourly_dipmac_marg_params_{margfamily}.csv"
        ),
        marg_params_daily_obs = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "lat_{lat}_lon_{lon}_ntr_daily_dipmac_marg_params_{margfamily}.csv"
        ),
        marg_params_daily_cmip = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_daily_dists_marg_{margfamily}_lat_{lat}_lon_{lon}.csv"
        )
    output:
        output_cmip_params_csv = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_cmip_params_marg_{margfamily}_lat_{lat}_lon_{lon}.csv"
        )
    script:
        "../scripts/run06a_dipmaccmipparams.py"


def dipmaccache_input(wc):
    """
    Return list of dipmaccmipparams output files based on the 
    discover_cmip_latlon_space checkpoint.
    """
    # Get the manifest dynamically from the checkpoint
    manifest_file = checkpoints.discover_cmip_latlon_space.get().output.manifest
    df = pd.read_csv(manifest_file, sep="\t")
    # Build the list of dipmaccmipparams outputs
    return [
        f"{BASEDIR}/05_cmip/{config['grid_params']}_"
        f"{config['obs_source_params']}_{config['dipmac_params']}/"
        f"{r.model}_{r.ripf}_{r.exp}/"
        f"ntr_cmip_params_marg_{wc.margfamily}_lat_{r.lat}_lon_{r.lon}.csv"
        for r in df.itertuples()
        # for margfamily in config['comboparams']['marginalfamilies']
        # for acsfamily in config['comboparams']['acsfamilies']
    ]


rule dipmaccacheactpnts:
    """Cache DiPMaC actpnts calculations (rounded to 2 decimal place)
    for the unique set of CMIP marginal distribution parameters
    to speed up computation.
    """
    input:
        dipmaccache_input,
        space_done = f"{BASEDIR}/00_metadata/cmip_space.done"
    output:
        actpnts = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actpnts_marg_{margfamily}.csv"
        ),
    script:
        "../scripts/run06b_dipmaccacheactpnts.py"


rule dipmaccachenkblocks:
    """Cache DiPMaC nkblocks calculations for the unique set of 
    CMIP marginal distribution parameters to speed up computation.
    """
    input:
        dipmaccache_input,
        space_done = f"{BASEDIR}/00_metadata/cmip_space.done"
    params:
        p_e = DIPMAC_CONFIG['p_e'],
        P = DIPMAC_CONFIG['P'],
    output:
        nkblocks = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_nkblocks_marg_{margfamily}.csv"
        ),
    script:
        "../scripts/run06b_dipmaccachenkblocks.py"


rule dipmaccacheactf:
    """Cache DiPMaC actf calculations for the unique set of
    actpnts to speed up computation.
    """
    input:
        actpnts = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actpnts_marg_{margfamily}.csv"
        ),
    output:
        actfpara = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actf_marg_{margfamily}.csv"
        ),
    script:
        "../scripts/run06c_dipmaccacheactf.py"


rule dipmacpicklecache:
    input:
        actf_cache_csv = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actf_marg_{margfamily}.csv"
        ),
        nkblocks_cache_csv = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_nkblocks_marg_{margfamily}.csv"
        )
    output:
        nkblocks_cache_pkl = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_nkblocks_marg_{margfamily}.pkl"
        ),
        actf_cache_pkl = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actf_marg_{margfamily}.pkl"
        )
    params:
        max_kblocks = DIPMAC_CONFIG['max_kblocks'],
    script:
        "../scripts/run06d_dipmacpicklecache.py"


rule dipmacrun:
    input:
        marg_params = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_cmip_params_marg_{margfamily}_lat_{lat}_lon_{lon}.csv"
        ),
        dipmac_actf_cache = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_actf_marg_{margfamily}.pkl"
        ),
        dipmac_nkblocks_cache = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "cache/cmip_dipmac_cache_nkblocks_marg_{margfamily}.pkl"
        ),
        acs_params_hourly_obs = (
            f"{BASEDIR}/04_params/dipmac_regression/"
            f"{config['grid_params']}/{config['obs_source_params']}/"
            f"{config['dipmac_params']}/"
            "lat_{lat}_lon_{lon}_ntr_hourly_dipmac_acs_params_{acsfamily}.csv"
        ),
        daily_ntr_cmip = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_daily_lat_{lat}_lon_{lon}.csv"
        )
    output: 
        hourly_ntr = (
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            "{model}_{ripf}_{exp}/"
            "ntr_hourly_lat_{lat}_lon_{lon}_marg_{margfamily}_acs_{acsfamily}.csv"
        )
    params:
        lagmax = DIPMAC_CONFIG['lagmax'],
        startyear = SRC_CONFIG['startyear']
    resources:
        mem_mb = 1000
    script:
        "../scripts/run06e_dipmac_run.py"


### Inputs for hourly files
def run06_inputs(_):
    # combine all the hourly outputs
    manifest_file = checkpoints.discover_cmip_latlon_space.get().output.manifest
    df = pd.read_csv(manifest_file, sep="\t")
    hourly_files = []
    monthly_files = []
    for r in df.itertuples():
        # monthly values
        monthly_files.append(
            f"{BASEDIR}/05_cmip/{config['grid_params']}_"
            f"{config['obs_source_params']}_{config['dipmac_params']}/"
            f"{r.model}_{r.ripf}_{r.exp}/msl_monthly.nc"
        )
        for margfamily in config['comboparams']['marginalfamilies']:
            for acsfamily in config['comboparams']['acsfamilies']:
                hourly_files.append(
                    f"{BASEDIR}/05_cmip/{config['grid_params']}_{config['obs_source_params']}_{config['dipmac_params']}/"
                    f"{r.model}_{r.ripf}_{r.exp}/"
                    f"ntr_hourly_lat_{r.lat}_lon_{r.lon}_marg_{margfamily}_acs_{acsfamily}.csv"
                )
    return hourly_files + monthly_files

## Generate all hourly files
rule run06:
    input:
        run06_inputs,
        space_done = f"{BASEDIR}/00_metadata/cmip_space.done"
    output:
        touch(f"{BASEDIR}/00_metadata/run06makesealevel.done")
