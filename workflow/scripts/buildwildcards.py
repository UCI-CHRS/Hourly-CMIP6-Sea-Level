
"""Get table of valid wildcards for CMIP model/ripf/exp and lat/lon coords."""

from pathlib import Path
import pandas as pd
from snakemake.script import snakemake


# Get CMIP MIDs that have a historical run
latlons = pd.read_csv(snakemake.input.latlons, usecols=["x", "y"]).drop_duplicates()
cmip_files = pd.read_csv(snakemake.input.upsampled)["file"]

lats = latlons['y']
lons = latlons['x']

rows = []
for f in cmip_files:
    name = Path(f).name
    model, ripf, exp = name.split("_")[:3]
    rows.append((model, ripf, exp))

all_mids = pd.DataFrame(
    rows,
    columns=["model", "ripf", "exp"]
)

# Check that the historical run is available
mids_with_historical = (
    all_mids.groupby(['model', 'ripf'])
    .apply(
        lambda df: any(['historical' in x for x in df.exp]),
        include_groups=False
    )
)
available_models_ripfs = (
    mids_with_historical[mids_with_historical
    ].reset_index()[['model', 'ripf']]
)

cmip = pd.merge(
    all_mids,
    available_models_ripfs, 
    left_on=('model', 'ripf'),
    right_on=('model', 'ripf'),
).drop_duplicates()

# Merge with latlon, margfamily, and acsfamily information for 
# monthly and ntr_hourly files

mids = cmip.drop_duplicates()
rows = []
for model, ripf, exp in zip(mids['model'], mids['ripf'], mids['exp']):
    for lat, lon in zip(lats, lons):
        rows.append((model, ripf, exp, lat, lon))

df = pd.DataFrame(
    rows,
    columns=["model", "ripf", "exp", "lat", "lon"]
)

df.to_csv(snakemake.output.manifest, sep="\t", index=False)
