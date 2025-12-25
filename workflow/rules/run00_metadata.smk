
rule makedirs:
    output: 
        str(makepath("metadata", "README.txt"))
    params:
        basepath = Path(config['BASE_DATA_PATH']),
        dirdict = config['dirdict'],
        grid_params = config['grid_params'],
        obs_source_params = config['obs_source_params'],
        dipmac_params = config['dipmac_params'],
        metadata_folder = makepath("metadata"),
        metadata_keys = config['metadata_keys'],
        parameter_descriptions_file = "config/metadata.yml"
    script:
        "../scripts/run00a_makedirs.py"


rule cmipmetadata:
    input: 
        str(makepath("metadata", "README.txt"))
    output: 
        [str(makepath("metadata", f"{exp}.csv"))
        for exp in config['comboparams']['cmipexperiments'].keys()]
    params:
        experiments = config['comboparams']['cmipexperiments'],
        metadata_folder = makepath("metadata")
    script:
        "../scripts/run00b_cmipmetadata.py"


rule oceangaugemetadata:  # station metadata for tide gauge observations
    input:
        str(makepath("metadata", "README.txt"))
    output:
        str(makepath("metadata", f"{SRC_CONFIG['obs_ocean_source']}_station_metadata.csv"))
    params:
        metadata_dir = makepath("metadata")
    script: 
        f"../scripts/run00c_ocean_{SRC_CONFIG['obs_ocean_source']}_metadata.py"


checkpoint metadata:
    input: 
        ocean = str(makepath("metadata", f"{SRC_CONFIG['obs_ocean_source']}_station_metadata.csv")),
        cmip = [
            str(makepath("metadata", f"{exp}.csv"))
            for exp in config['comboparams']['cmipexperiments'].keys()
        ]
    output:
        ocean = f"{BASEDIR}/00_metadata/ocean_metadata_manifest.csv",
        cmip = f"{BASEDIR}/00_metadata/cmip_metadata_manifest.csv"
    run:
        # Copy files
        cmip = pd.concat([pd.read_csv(f) for f in input.cmip]).drop_duplicates()
        ocean = pd.read_csv(input.ocean)
        cmip.to_csv(output.cmip)
        ocean.to_csv(output.ocean)
