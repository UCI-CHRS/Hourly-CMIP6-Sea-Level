
"""Download CMIP, ERA5, and NOAA data. Then check for missing data 
and make the CMIP reference grids.
"""
###---------------------------------------------------------------------
### Download raw data 
###---------------------------------------------------------------------

rule cmipraw_onemid:
    input: 
        [str(makepath("00_metadata", f"{exp}.csv"))
        for exp in config['comboparams']['cmipexperiments'].keys()]
    output: 
        logfile = f"{BASEDIR}/00_metadata/logs/01_rawdata/cmip_{{mid}}.log"
    params:
        mid = lambda wc: wc.mid,
        ncdir = makepath("rawdata", "cmip", "nc")
    script:
        "../scripts/run01a_cmip6raw.py"


rule obsatmosraw_peryear:  # so far only implemented for ERA5
    input: 
        str(makepath("00_metadata", "README.txt")),
        f"{config['HOME_DIRECTORY']}/.cdsapirc"
    output: 
        touch(f"{BASEDIR}/00_metadata/touchfiles/touchfile_obs_atmos_"
              f"{SRC_CONFIG['obs_atmos_source']}_{{year}}.done")
    params:
        data_dir = makepath("01_rawdata", "obs", "atmos", SRC_CONFIG['obs_atmos_source']),
        # We want to download for the CONUS; all other bboxes are a subset of 
        # that and don't warrant a re-download. 
        bbox_conus = config['metadata_keys']['conus'],
        year = lambda wc: wc.year,
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear']
    script:
        f"../scripts/run01a_atmos_{SRC_CONFIG['obs_atmos_source']}_raw.py"


rule obsoceanraw_perstation:  # So far only implemented for NOAA
    input: 
        str(makepath("00_metadata", f"{SRC_CONFIG['obs_ocean_source']}_station_metadata.csv"))
    wildcard_constraints:
        stn = r"\d{7}"
    output: 
        touch(str(
            makepath("00_metadata", "touchfiles", 
            f"touchfile_obs_ocean_{SRC_CONFIG['obs_ocean_source']}_"
            "{stn}.done")
        ))
    params:
        raw_data_dir = makepath("01_rawdata", "obs", "ocean", SRC_CONFIG['obs_ocean_source']),
        stn = lambda wc: wc.stn,
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear']
    script:
        f"../scripts/run01a_ocean_{SRC_CONFIG['obs_ocean_source']}_raw.py"


###---------------------------------------------------------------------
### Check for missing or invalid data
###---------------------------------------------------------------------

rule checkcmip: 
    input:
        logfile = f"{BASEDIR}/00_metadata/logs/01_rawdata/cmip_{{mid}}.log"
    output:
        available = f"{BASEDIR}/00_metadata/logs/01_rawdata/cmip_{{mid}}_available.csv",
        missing = f"{BASEDIR}/00_metadata/logs/01_rawdata/cmip_{{mid}}_missing.csv"
    params:
        mid = lambda wc: wc.mid,
        cmip_nc_dir = makepath("01_rawdata", "cmip", "nc"),
    script:
        "../scripts/run01b_checkcmipdownloads.py"


rule checkatmos:
    input: 
        f"{BASEDIR}/00_metadata/touchfiles/touchfile_obs_atmos_"
        f"{SRC_CONFIG['obs_atmos_source']}_{{year}}.done"
    output:
        available = (
            f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_atmos_source']}"
            "_hourly_{year,\d{4}}_available.csv"
        ),
        missing = (
            f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_atmos_source']}"
            "_hourly_{year,\d{4}}_missing.csv"
        )
    params:
        atmos_nc_dir = makepath(
            "rawdata", "obs", "atmos", SRC_CONFIG['obs_atmos_source']
        ),
        fnames = [
            f"{SRC_CONFIG['obs_atmos_source']}_hourly_wind_slp_{{year}}_"
            f"{str(month).zfill(2)}.nc"
            for month in range(1, 13)
        ],
    script:
        f"../scripts/run01b_check{SRC_CONFIG['obs_atmos_source']}downloads.py"


rule checkocean: 
    wildcard_constraints:
        stn = r"\d{7}"
    input: 
        f"{BASEDIR}/00_metadata/touchfiles/touchfile_obs_ocean_"
        f"{SRC_CONFIG['obs_ocean_source']}_{{stn}}.done"
    output: 
        available = (
            f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_ocean_source']}_"
            "{stn}_available.csv"
        ),
        missing = (
            f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_ocean_source']}_"
            "{stn}_missing.csv"
        )
    params:
        raw_data_dir = makepath(
            "01_rawdata", "obs", "ocean", SRC_CONFIG['obs_ocean_source']
        ),
        stn = lambda wc: wc.stn, 
        sy = SRC_CONFIG['startyear'],
        ey = SRC_CONFIG['endyear'],
    script: 
        f"../scripts/run01b_check{SRC_CONFIG['obs_ocean_source']}downloads.py"


###---------------------------------------------------------------------
### Make the CMIP reference grid
###---------------------------------------------------------------------

checkpoint makecmiprefgrids:
    input:
        cmip_mids = f"{BASEDIR}/00_metadata/cmip_metadata_manifest.csv",
        tide_gauges = f"{BASEDIR}/00_metadata/ocean_metadata_manifest.csv",
    output:
        latlons = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}_"
            f"ngrids_{GRD_CONFIG['ngrids_window']}_"
            f"{SRC_CONFIG['obs_ocean_source']}stns.csv"
        ),
        refgrid = (
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        )
    params:
        upsampled_grids_dir = makepath("upsampleddata", "grids"),
        rawdata_cmip_nc_dir = makepath("rawdata", "cmip", "nc"),
        ngrids = GRD_CONFIG['ngrids_window'],
        grid_id = GRD_CONFIG['gridname']
    script:
        "../scripts/run01c_makecmiprefgrids.py"


###---------------------------------------------------------------------
### Summarize missing data and available files for obs, atmos, and CMIP
###---------------------------------------------------------------------

def rawdata_input_cmip(_):
    """Forces checkdownloads to use the CMIP metadata file generated
    in an earlier rule. 
    """
    checkpoint_output = checkpoints.metadata.get().output['cmip']
    mid_list = pd.read_csv(checkpoint_output)['id'].tolist()
    mids = list({mid.split('|')[0][:-10] for mid in mid_list})
    return [
        f"{BASEDIR}/00_metadata/logs/01_rawdata/cmip_{mid}_{status}.csv"
        for mid in mids for status in ['available', 'missing']
    ]

def rawdata_input_ocean(_):
    """Forces checkdownloads to use the tide station metadata file generated
    in an earlier rule. 
    """
    checkpoint_output = checkpoints.metadata.get().output['ocean']
    stations = pd.read_csv(
        f"{BASEDIR}/00_metadata/{SRC_CONFIG['obs_ocean_source']}_station_metadata.csv"
    )['id']
    return [
        f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_ocean_source']}_"
        f"{stn}_{status}.csv"
        for stn in stations for status in ['available', 'missing']
    ]

checkpoint rawdata:
    input:
        cmip_refgrids = [
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}_"
            f"ngrids_{GRD_CONFIG['ngrids_window']}_"
            f"{SRC_CONFIG['obs_ocean_source']}stns.csv",
            f"{BASEDIR}/03_upsampleddata/grids/{GRD_CONFIG['gridname']}.nc"
        ],
        cmip = rawdata_input_cmip,
        atmos = [
            f"{BASEDIR}/00_metadata/logs/01_rawdata/{SRC_CONFIG['obs_atmos_source']}"
            f"_hourly_{year}_{status}.csv"
            for year in range(SRC_CONFIG['startyear'], SRC_CONFIG['endyear']+1)
            for status in ['available', 'missing']
        ],
        ocean = rawdata_input_ocean,
    output:
        cmip_available = f"{BASEDIR}/00_metadata/cmip_rawdata_manifest.csv",
        cmip_missing = (
            f"{BASEDIR}/00_metadata/cmip_rawdata_"
            f"{'_'.join(config['comboparams']['cmipexperiments'].keys())}"
            "_missing.csv"
        ),
        atmos_available = f"{BASEDIR}/00_metadata/atmos_rawdata_manifest.csv",
        atmos_missing = f"{BASEDIR}/00_metadata/atmos_{SRC_CONFIG['obs_atmos_source']}_rawdata_missing.csv",
        ocean_available = f"{BASEDIR}/00_metadata/ocean_rawdata_manifest.csv",
        ocean_missing = f"{BASEDIR}/00_metadata/ocean_{SRC_CONFIG['obs_ocean_source']}_rawdata_missing.csv",
    script:
        "../scripts/run01d_rawdatasummary.py"
