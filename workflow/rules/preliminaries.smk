import os
from pathlib import Path
import yaml
import pandas as pd

configfile: "config/config.yml"
configfile: "config/directories.yml"
configfile: "config/metadata.yml"

# Get parameter sets from config file
SRC_CONFIG = config['metadata_keys'][config['obs_source_params']]
GRD_CONFIG = config['metadata_keys'][config['grid_params']]
DIPMAC_CONFIG = config['metadata_keys'][config['dipmac_params']]
BASEDIR = config['BASE_DATA_PATH']
EMISSION_SCENARIOS = sorted({
    x for y in (
        config['comboparams']['cmipexperiments'][k]
        for k in config['comboparams']['cmipexperiments']
    )
    for x in y
})
BBOX = config['metadata_keys'][SRC_CONFIG['bbox']]
BBOX_CONUS = config['metadata_keys']['conus']

SCRIPTDIR = Path(__file__).parent / "scripts"
