
"""Set of general utilities, mainly for dealing with directories and file paths."""

from functools import wraps
import time
import os
import pathlib
from collections.abc import Callable
import datetime as dt

import numpy as np
import pandas as pd


def timing(f: Callable) -> Callable:
    """Decorator function to time another function.

    Args:
        f (Callable):
            function to wrap

    Returns:
        Callable:
            Timing decorator
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'Function {f.__name__!r} executed in {(te-ts):.4f}s')
        return result
    return wrap


def flatten_list(l: list) -> list:
    """Flattens a list of lists.

    Args:
        l (list):
            list of lists

    Returns:
        list:
            l with first level of nesting flattened
    """
    return [item for sublist in l for item in sublist]


def make_directory_tree(dirlist) -> None:
    """Creates directory tree structure specified in dir_dict.

    Args:
        dir_dict (dict[pathlib.Path]):
            dictionary of named paths to create.

    Returns:
        None
    """
    sortidx = [str(p).count("/") for p in dirlist]
    dirlist_sorted = [x for _, x in sorted(zip(sortidx, dirlist))]
    for p in dirlist_sorted:
        p.mkdir(parents=True, exist_ok=True)


def get_relative_path(dir: pathlib.Path, base_path: pathlib.Path) -> pathlib.Path:
    """Returns the absolute path dir, minus the prefix base_path.

    Example:
        dir = "/a/b/c/d"
        base_path = "/a/b"
        get_relative_path(dir, base_path) returns "c/d"

    Args:
        dir (pathlib.Path):
            absolute path
        base_path (pathlib.Path):
            prefix path to dir to remove

    Returns:
        pathlib.Path:
            dir with base_path prefix removed
    """
    relative_path = dir.split(base_path)[1]
    if relative_path[0] == "/":
        relative_path = relative_path[1:]
    return relative_path


def mirror_relative_subpaths(base_path: pathlib.Path) -> list[pathlib.Path]:
    """Get all relative pathnames descended from base_path.

    Args:
        base_path (pathlib.Path):
            parent directory

    Returns:
        list[pathlib.Path]:
            list of relative pathnames under base_path
    """
    dirs = [f[0] for f in os.walk(base_path)]
    return [get_relative_path(dir, base_path) for dir in dirs]


def mirror_files(files: list[pathlib.Path],
                 ref_base_path: pathlib.Path,
                 new_base_path: pathlib.Path) -> list[pathlib.Path]:
    """Mirror the directory structure of ref_base_path in new_base_path.

    Args:
        files (list[pathlib.Path]):
            list of files for which to change parent directory
        ref_base_path (pathlib.Path):
            parent directory from which to get relative subpaths
        new_base_path (pathlib.Path):
            new base path prefix to apply to the relative subpaths

    Returns:
        list[pathlib.Path]:
            new absolute paths
    """
    return [pathlib.Path(f.replace(str(ref_base_path), str(new_base_path))) for f in files]


def file_doesnt_exist(filename: pathlib.Path) -> bool:
    """To enable only running a function if a certain filename
    doesn't yet exist.

    Args:
        filename (pathlib.Path):
            path for which to test existence

    Returns:
        bool:
            True if the file doesn't exist, False if it does
    """
    return not filename.exists()


def time_steps(interval: dt.timedelta) -> list[dt.datetime]:
    """Get the time steps for one (leap) year given interval (probably days)

    Args:
        interval (dt.timedelta):
            time interval (e.g., 1 day)

    Returns:
        list[dt.datetime]:
            list of datetimes at the specified interval for a sample leap year.
    """
    leap_year = 1980  # use a leap year
    d1 = dt.datetime(leap_year, 1, 1, 0, 0, 0)
    d2 = dt.datetime(leap_year, 12, 31, 23, 59, 59)
    n_intervals = int(np.ceil((d2 - d1)/interval))
    ts = [d1 + ind * interval for ind in range(n_intervals)]
    return ts


def progress_bar(progress, total, offset=""):
    percent = 100 * (progress / float(total))
    bar = '█' * int((percent)/2) + '-' * int((100 - percent)/2)
    print(f"\r{offset}|{bar}| {percent:.2f}%", end="\r")
