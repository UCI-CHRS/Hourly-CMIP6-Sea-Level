# Historical and future CMIP6 hourly coastal water levels in the United States 

This repository contains all code used to generate the dataset "Historical and future CMIP6 hourly coastal water levels in the United States" for the publication "Historical and future coastal water levels in the United States for impact-based risk and attribution analysis" (Hjelmstad, A. et al. 2026, under review in Scientific Data). All data sources, methods, and usage examples will be available upon publication. 

The final dataset is hosted on Dryad at DOI: 10.5061/dryad.5dv41nskn

## Abstract
Coastal flooding has emerged as an increasingly costly and deadly compound hazard. Although flood risk assessment is crucial for minimizing such losses, analyses of impact-based flood hazard scenarios are prone to a mismatch between the fine spatiotemporal scales at which the worst flooding impacts occur (on the order of hours and kilometers), and the coarse scales at which global climate model outputs are available (daily or monthly over hundreds of kilometers). To address one of these scale problems, we present here hourly sea level estimates for several CMIP6 emissions scenarios to better capture sub-daily extremes in sea level that can exacerbate coastal flooding during hurricanes and tropical cyclones. The dataset includes estimates of hourly total water level along the contiguous U.S. coasts in addition to the component mean sea level, tide, and non-tidal residual values. The resulting high resolution, physically consistent scenarios of coastal water level extremes enable linking global emissions scenarios to local flooding extremes via flood models for a range of future and historical scenarios. 

## Usage

All code was written in Python and is configured to run in a Snakemake workflow. 

### Prerequisites

The conda environment specified in workflow/envs/environment.yml should be activated for running all code. Note that the scripts that run the MAGICC simple climate model use the MAGICC6 executable bundled with the `pymagicc` module, which may behave differently on different machines. For this step, running in a Docker or Singularity environment using the recipe at workflow/envs/Dockerfile is recommended.

The run specifications in each of the config/ files reflects what is in the dataset, except the following variables should be changed:

- `BASE_DATA_PATH`: Path to all final and intermediate data files. Should be a disk with several TB of available space. 
- `HOME_DIRECTORY`: Local machine's home directory. This is only needed for referencing the .cdsapirc file needed to download ERA5 reanalysis data from the Copernicus Climate Data Store. You will have to generate your own API key for that. Instructions to do so are [here](https://cds.climate.copernicus.eu/how-to-api).

### Workflow
The base Snakefile outlines the full analysis pipeline, although separate steps are provided in Snakefile_01 through Snakefile_05 to enable running some steps on an HPC. Each subsequent Snakefile assumes all outputs in preceding Snakefiles have already been generated. 

### Resource requirements
Some steps in the analysis pipeline are only feasibly run on an HPC. For these cases, resources have been specified per snakemake rule, and are configured to run as relatively lightweight parallel jobs. 

### Figures
Code to generate all figures in the corresponding paper are in the figures/ folder. They use absolute file paths, so change them to match the location of generated data on your machine. 

## Contact

Corresponding author: Annika Hjelmstad (ahjelmst@uci.edu)

Project Link: [https://github.com/UCI-CHRS/Hourly-CMIP6-Sea-Level](https://github.com/UCI-CHRS/Hourly-CMIP6-Sea-Level)
