"""Touch missing directories and write metadata from config files"""

from pathlib import Path
import shutil
from collections.abc import Generator
import yaml
from snakemake.script import snakemake


def path_from_dict(
    nested_path_dict: dict[str, dict | None]
    ) -> Generator[str, None, None]:
    """Generator function to create a list of relative paths 
    from a nested dictionary.
    
    Args:
        nested_path_dict (dict[str,str]): 
            nested dictionary containing path structure of the form:
            {'dir1': 
                'dir1_1':
                    'dir1_1_1': None,
                'dir1_2': None,
             'dir2': 
                'dir2_1': None,
            }

    Returns:
        Generator[str, None, None]:
            generator that yields a string representation of a path
    """
    for key, value in nested_path_dict.items():
        if isinstance(value, dict):
            for subkey in path_from_dict(value):
                yield f"{key}/{subkey}"
        else:
            yield key

def makedirs(
    basepath: Path,
    dirdict: dict[str, dict | None]
    ) -> None:
    """Touch all missing directories from the nested directory structure
    described in path. 

    Args:
        basepath (Path): Absolute file path to make the relative file 
            structure from.
        dirdict (dict[str, dict|None]): nested dictionary of string paths; 
            see example in path_from_dict() docstrings. 
    
    Returns:
        None: creates missing directories at basepath; 
            no function output
    """
    paths = list({
        basepath.joinpath(path) for sublist in [
            [Path(x)] + [p for p in Path(x).parents if p != Path('.')]
            for x in path_from_dict(dirdict)
        ] for path in sublist
    })
    paths_sorted = sorted(paths, key=lambda p: len(p.parts))
    for path in paths_sorted:
        path.mkdir(parents=True, exist_ok=True)


def writemetadata(
    grid_params: str,
    obs_source_params: str,
    dipmac_params: str,
    metadata_folder: Path,
    metadata_keys: dict[str, dict],
    parameter_descriptions_file: str
    ) -> None:
    """Write metadata for grouped parameters.

    Args:
        grid_params (str): grid_params metadata key (grdXXX)
        obs_source_params (str): obs_source_params metadata key (srcXXX)
        dipmac_params (str): dipmac_params metadata key (dipmacXXX)
        metadata_folder (Path): parent directory for metadata files
        metadata_keys (dict[str, dict]): dictionary of all key: metadata pairs
        parameter_descriptions (dict[str, dict]): file with parameter descriptions
            to copy over to base path. 

    Returns:
        None: Writes metadata files; no function output
    """
    shutil.copy(parameter_descriptions_file, metadata_folder.joinpath("README.txt"))
    with open(metadata_folder.joinpath(f"{grid_params}.txt"),
        "w", encoding="utf-8") as f:
        f.write(f"{yaml.dump(metadata_keys[grid_params])}")
    with open(metadata_folder.joinpath(f"{obs_source_params}.txt"),
        "w", encoding="utf-8") as f:
        f.write(f"{yaml.dump(metadata_keys[obs_source_params])}")
    with open(metadata_folder.joinpath(f"{dipmac_params}.txt"),
        "w", encoding="utf-8") as f:
        f.write(f"{yaml.dump(metadata_keys[dipmac_params])}")


def main():
    """Main script to run via snakemake."""
    #-Snakemake params, inputs, outputs---------------------------------
    params = snakemake.params
    #-Script------------------------------------------------------------
    makedirs(
        basepath = params['basepath'],
        dirdict = params['dirdict']
    )
    writemetadata(
        grid_params = params['grid_params'],
        obs_source_params = params['obs_source_params'],
        dipmac_params = params['dipmac_params'],
        metadata_folder = params['metadata_folder'],
        metadata_keys = params['metadata_keys'],
        parameter_descriptions_file = params['parameter_descriptions_file']
    )


if __name__ == "__main__":
    main()
