
"""Utility functions for working with MAGICC. """

import glob
import os
from dataclasses import dataclass
import re
from io import StringIO
import shutil
from functools import reduce
from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
import pymagicc


def debug_input_files(rundir: Path) -> None:
    """Copy MAGICC's in files from tmp to an accessible folder to inspect

    Args:
        rundir (Path):
            temporary run directory withe MAGICC in files

    Returns:
        None (copied direclty to disk)
    """
    shutil.copytree(rundir, "/workspaces/cmip6-hourly-sea-level/test")


def emission_infiles(source_dir: Path,
                     infile_patterns: list[str] = ["*IN", "*.prn"],
                     exceptions: list[str] = None) -> list[str]:
    """Glob all the emission .in files in source_dir.

    Args:
        source_dir (Path):
            directory containing the .in files
        infile_patterns (list[str]):
            strin gpattersn for files to include
        exceptions (list[str]):
            string patterns for files to ignore

    Returns:
        list[str]:
            list of .in files in source_dir
    """
    infiles = [list(source_dir.glob(x)) for x in infile_patterns]
    if exceptions is not None:
        infiles = [x for x in infiles if not any(e in x for e in exceptions)]
    return reduce(lambda q, p: p+q, infiles)


def get_infile_header(file: Path) -> tuple[str, str]:
    """Split a .in file into the header and body.

    Args:
        file (Path):
            .in file to split

    Returns:
        tuple[str, str]:
            header and body strings.
    """
    with open(file, "r") as f:
        txt = f.read()
    parsed = re.search("\n\s{3,}[1]", txt)
    header = txt[:parsed.start()]
    body = txt[parsed.start():]
    return header, body


def detect_fwf_widths(body: str) -> list[int]:
    """figure out widths of fixed with txt file.

    Args:
        body (str):
            body text (excluding header) of txt file

    Returns:
        list[int]:
            widths of each fixed-width column.
    """
    first_line = body.split("\n")[1]
    starts = [match.start() for match in re.finditer("[\s]{4,}", first_line)]
    widths = np.diff(starts).tolist() + [len(first_line) - starts[-1]]
    return widths


def read_fwf(body: str) -> pd.DataFrame:
    """Detect widths from a file split into header and body components
        and return data as a pd.DataFrame

    Args:
        body (str):
            body text (excluding header) of txt file

    Returns:
        pd.DataFrame:
            data table in the fixed with file
    """
    widths = detect_fwf_widths(body)
    df = pd.read_fwf(StringIO(body), header=None, widths=widths)
    df.set_index(df.columns[0], inplace=True)
    return df


def read_in(file: Path) -> pd.DataFrame:
    """Read a .in file.

    Args:
        file (Path):
            .in file to read

    Returns:
        pd.DataFrame:
            data table from the .in file
    """
    _, body = get_infile_header(file)
    df = read_fwf(body)
    return df


def get_prn_widths(body: str) -> list[int]:
    """Get fixed widths from .prn files.

    Args:
        body (str):
             body text (excluding header) of the .prn file

    Returns:
        list[int]:
            widths of each fixed-width column.
    """
    first_line = body.split("\n")[4]
    starts = [match.start() for match in re.finditer("[\s]{3,}", first_line)]
    widths = np.diff(starts).tolist() + [len(first_line) - starts[-1]]
    if len(widths) < 2:
        starts = [match.start()
                  for match in re.finditer("[\s]{1,}", first_line)]
        widths = np.diff(starts).tolist() + [len(first_line) - starts[-1]]
    return widths


def _parse_prn(file: Path) -> tuple[str, str]:
    """Split a .prn file into the header and body.

    Args:
        file (Path):
            .in file to split

    Returns:
        tuple[str, str]:
            header and body strings.
    """
    with open(file, "r") as f:
        txt = f.read()
    parsed = re.search("['CH3Cl']\n{2,}", txt)
    header = txt[:parsed.start()] + "l\n\n"
    body = txt[parsed.start():].split("l\n\n")[1]
    # Remove post-data comments
    comments = re.search("\n{2,}", body)
    if comments is not None:
        body = body[:comments.start()]
    return header, body


def read_prn(file: Path) -> pd.DataFrame:
    """Read a .prn file.

    Args:
        file (Path):
            .prn file to read

    Returns:
        pd.DataFrame:
            data table from the .prn file
    """
    _, body = _parse_prn(file)
    widths = get_prn_widths(body)
    df = pd.read_fwf(StringIO(body), header=None, widths=widths)
    df.set_index(df.columns[0], inplace=True)
    df.index = df.index.astype(dtype=int)
    return df


@dataclass
class infile:
    """Read, parse, and modify standard MAGICC .IN files during a pyMAGICC run.

    Attributes:
        magicc (pymagicc.core.MAGICC7 | pymagicc.core.MAGICC6):
            the magic run object
        fname (Path):
            .in or .prnfile name
    """
    magicc: pymagicc.core.MAGICC7 | pymagicc.core.MAGICC6
    fname: Path

    @property
    def rundir(self) -> str:
        """MAGICC run directory.

        Args:

        Returns:
            str:
                MAGICC run directory
        """
        return Path(self.magicc.run_dir)

    def read(self) -> pd.DataFrame:
        """Read data from a .in or .prn file.

        Args:

        Returns:
            pd.DataFrame:
                data table from the .in or .prn file
        """
        ext = self.fname.suffix
        if ext == ".IN":
            return read_in(self.rundir.joinpath(self.fname))
        elif ext == ".prn":
            return read_prn(self.rundir.joinpath(self.fname))

    def rewrite_in(self, replacement_df: pd.DataFrame, newpath: str,
                   source_dir: str = None) -> str:
        """ Edit the .IN files in the temporary run directory.

        Args:
            replacement_df (pd.DataFrame):
                the df to replace in the tmp directory.
            newpath (str):
                directory to save file to
            source_dir (str | None):
                directory containing self.fname. Set to self.rundir if
                not supplied

        Returns:
            str:
                new pathname for the self.fname file in the temporary
                run directory
        """
        if source_dir is None:
            source_dir = self.rundir
        header, body = get_infile_header(source_dir.joinpath(self.fname))
        widths = detect_fwf_widths(body)
        precision = [".0f"] + [".5e"] * (len(widths) - 1)
        fmt = " ".join([f"%{str(w)}{p}" for w, p in zip(widths, precision)])
        np.savetxt(newpath, replacement_df.reset_index().values,
                   fmt=fmt, header="".join(header), comments="")
        return newpath

    def rewrite_prn(self, replacement_df: pd.DataFrame, newpath: str,
                    source_dir: str = None) -> str:
        """Edit the .prn files in the temporary run directory.

        Args:
            replacement_df (pd.DataFrame):
                the df to replace in the tmp directory.
            newpath (str):
                directory to save file to
            source_dir (str | None):
                directory containing self.fname. Set to self.rundir if
                not supplied

        Returns:
            str:
                new pathname for the self.fname file in the temporary
                run directory
        """
        if source_dir is None:
            source_dir = self.rundir
        header, _ = _parse_prn(self.fname)
        if "RCPODS_WMO2006_MixingRatios_A1.prn" in self.fname.name:
            N = replacement_df.shape[1]
            fmt = ' '.join(['%i'] + ['%0.3e']*N)
        else:
            # " ".join([f"%{str(w)}{p}" for w,p in zip(widths, precision)])
            fmt = "%10.0f"
        np.savetxt(newpath, replacement_df.reset_index().values,
                   fmt=fmt, header="".join(header), comments="")
        return newpath


def propagate_baseline_emissions(magicc: pymagicc.core.MAGICC7 | pymagicc.core.MAGICC6,
                                 fname: Path, outdir: Path,
                                 summary_fun: Callable = np.median,
                                 start_year: int = 1850, end_year: int = 1909
                                 ) -> pd.DataFrame:
    """Extend historical emissions based on a baseline period for one
        historical emission .IN file.
        NOTE: the default start and end years (1850-1909) correspond
        to the baseline period used to define the hist-nat scenario
        in CMIP6.

    Args:
        pymagicc.core.MAGICC7 | pymagicc.core.MAGICC6:
            magicc run object
        fname (Path):
            filename of the historical emission .IN file
        outdir (Path):
            directory containing filename
        summary_fun (Callable):
            function of the baseline period to use in propagating forward.
            Defaults to setting all post-baseline values to the median
            during the start to end year period.
        start_year (int):
            first year of baseline period
        end_year (int):
            last year of baseline period

    Returns:
        pd.DataFrame:
            extended emissions time series
    """
    my_infile = infile(magicc, fname)
    df = my_infile.read()  # .astype(float)
    if df.index.min() < end_year:
        baseline_emissions = df.loc[start_year:end_year]
        baseline_emissions = baseline_emissions.astype(float)
        col_summary = baseline_emissions.apply(summary_fun)
        df.loc[end_year:, :] = col_summary.values
    else:  # if the baseline period doesn't exist, set all to zero.
        df.loc[:, :] = df.iloc[0, :].values * 0
    newpath = outdir.joinpath(fname.name)
    ext = fname.suffix
    if ext == ".IN":
        my_infile.rewrite_in(replacement_df=df, newpath=newpath)
    elif ext == ".prn":
        my_infile.rewrite_prn(replacement_df=df, newpath=newpath)
    return df


def make_ssp_scen_files(rawdata_cmip_sspemissions_path: Path):
    """Data from the IIASA SSP scenario database at.
    It needs to be reformatted to follow the MAGICC .SCEN file format.
    """
    for ssp in (119, 126, 245, 370, 434, 585):
        scen_path = rawdata_cmip_sspemissions_path.joinpath(f"SSP{ssp}.SCEN")
        if not scen_path.exists():
            df = [pd.read_csv(x) for x in
                  rawdata_cmip_sspemissions_path.glob(f"magicc/SSP{ssp}*.csv")
                  ][0]
            years = list(set(list(df.columns)).difference(set(('data_id', 'model', 'reference_period_end_year',
                         'reference_period_start_year', 'region', 'scenario', 'stage', 'todo', 'unit', 'variable'))))
            df_mod = (pd.melt(df.rename({"variable": "VARIABLE"}, axis=1), value_vars=years,
                              id_vars=('data_id', 'model', 'reference_period_end_year', 'reference_period_start_year', 'region', 'scenario', 'stage', 'todo', 'unit', 'VARIABLE'))
                      .rename({"variable": "year"}, axis=1)
                      .drop(['data_id', 'reference_period_start_year', 'reference_period_end_year', 'todo'], axis=1)
                      .loc[(lambda df: df.stage == "clean")]
                      .replace({
                          'Emissions|CO': "CO",  # units - MtCO
                          'Emissions|OC': "OC",  # Mt
                          'Emissions|VOC': "NMVOC",  # Mt
                          'Emissions|C2F6': "C2F6",  # kt
                          'Emissions|C6F14': "C6F14",  # kt
                          'Emissions|CF4': "CF4",  # kt
                          'Emissions|HFC125': "HFC125",  # kt
                          'Emissions|HFC134a': "HFC134a",  # kt
                          'Emissions|HFC143a': "HFC143a",  # kt
                          'Emissions|HFC227ea': "HFC227ea",  # kt
                          'Emissions|HFC23': "HFC23",  # kt
                          'Emissions|HFC245fa': "HFC245fa",  # kt
                          'Emissions|HFC32': "HFC32",  # kt
                          'Emissions|HFC4310mee': "HFC43-10",  # kt
                          'Emissions|SF6': "SF6",  # kt
                          'Emissions|Sulfur': "SOx",  # MtS
                          'Emissions|NH3': "NH3",  # MtN
                          'Emissions|NOx': "NOx",  # MtN
                          'Emissions|CO2|MAGICC Fossil and Industrial': "FossilCO2",  # GtC
                          'Emissions|N2O': "N2O",  # MtN2O-N
                          'Emissions|CH4': "CH4",  # MtCH4
                          'Emissions|BC': "BC",  # Mt
                          'Emissions|CO2': "OtherCO2",  # GtC
                      })
                      .loc[lambda df: ["Emissions|" not in col for col in df.VARIABLE]]
                      )
            df_scen = pd.pivot_table(df_mod,
                                     index='year',
                                     columns="VARIABLE",
                                     values="value"
                                     ).reset_index()
            # Convert units
            df_scen['OtherCO2'] = (
                df_scen['OtherCO2'] - df_scen['FossilCO2']) / 1000
            df_scen['FossilCO2'] = df_scen['FossilCO2'] * 12 / 44000
            df_scen['N2O'] = df_scen['N2O'] * 28 / (28 + 16) / 1000
            df_scen['SOx'] = df_scen['SOx'] * 32 / 64
            df_scen['NOx'] = df_scen['NOx'] * 14 / 30
            df_scen['NH3'] = df_scen['NH3'] * 14 / 17

            # Write to .SCEN file
            header = (
                f" {len(df_scen.year.unique())}\n"
                " 11\n"
                f" SSP{ssp}\n"
                " Description line 1\n"
                " Description line 2\n\n"
                " WORLD\n"
                "      YEARS  FossilCO2   OtherCO2        CH4        N2O        SOx         CO      NMVOC        NOx         BC         OC        NH3        CF4       C2F6      C6F14      HFC23      HFC32   HFC43-10     HFC125    HFC134a    HFC143a   HFC227ea   HFC245fa        SF6\n"
                "        Yrs        GtC        GtC      MtCH4    MtN2O-N        MtS       MtCO         Mt        MtN         Mt         Mt        MtN         kt         kt         kt         kt         kt         kt         kt         kt         kt         kt         kt         kt"
            )

            df_scen = df_scen[["year", "FossilCO2", "OtherCO2", "CH4", "N2O",
                               "SOx", "CO", "NMVOC", "NOx", "BC", "OC", "NH3",
                               "CF4", "C2F6", "C6F14", "HFC23", "HFC32", "HFC43-10",
                               "HFC125", "HFC134a", "HFC143a", "HFC227ea", "HFC245fa", "SF6"]]
            df_scen['year'] = df_scen['year'].astype(int)
            np.savetxt(scen_path, df_scen.values, fmt=' '.join(
                ['%11i'] + ['%10.4f']*23), header=header, footer="\n\n", comments="")


def ssp_scen_files(ssp: int, rawdata_cmip_sspemissions_path: Path):
    scen_path = rawdata_cmip_sspemissions_path.joinpath(f"SSP{ssp}.SCEN")
    if not scen_path.exists():
        make_ssp_scen_files(rawdata_cmip_sspemissions_path)
    scen = pymagicc.scenarios.read_scen_file(str(scen_path))
    return scen
