#!/usr/bin/env python3
"""
Run a Snakemake script outside Snakemake by injecting a mock snakemake object.

Supports:
- JSON config file (--config-file)
- CLI list-style I/O (--input foo bar)
- CLI dict-style I/O (--input ref=ref.fa reads=reads.fq)

Example from JSON config:
python workflow/aux/wrapsnakemake.py workflow/scripts/my_script.py \ 
    --config-file debug_case.json

Example from CLI arguments (list-style):
python workflow/aux/wrapsnakemake.py workflow/scripts/my_script.py \
    --input data/input1.txt data/input2.txt \
    --output results/output.txt \
    --param size=42 label=debug_run

Example from CLI arguments (dict-style):
python workflow/aux/wrapsnakemake.py workflow/scripts/my_script.py \
    --input ref=data/ref.fa reads=data/sample.fq \
    --output bam=results/sample.bam log=results/sample.log \
    --param size=42 label=debug_run \
    --wildcard sample=S1
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def parse_kv_or_list(pairs):
    """Parse CLI args that may be list or key=value."""
    if not pairs:
        return []
    if all("=" in p for p in pairs):
        return {k: v for k, v in (p.split("=", 1) for p in pairs)}
    return pairs


def make_snakemake(config):
    """Create a dummy snakemake object from dict (JSON or CLI-parsed)."""
    return SimpleNamespace(
        input=config.get("input", []),
        output=config.get("output", []),
        params=config.get("params", {}),
        wildcards=config.get("wildcards", {}),
        log=config.get("log", []),
        threads=config.get("threads", 1),
        resources=config.get("resources", {}),
        config=config.get("config", {})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Snakemake script standalone.")
    parser.add_argument("script", help="Path to the Snakemake script")
    parser.add_argument("--config-file", help="JSON config with I/O/04_params/wildcards")

    # Optional CLI-based config
    parser.add_argument("--input", nargs="*", help="Input files (list or key=val)")
    parser.add_argument("--output", nargs="*", help="Output files (list or key=val)")
    parser.add_argument("--param", nargs="*", help="Params in key=value form")
    parser.add_argument("--wildcard", nargs="*", help="Wildcards in key=value form")

    args = parser.parse_args()

    # Ensure workflow root is in sys.path
    script_path = Path(args.script).resolve()
    workflow_root = script_path.parent.parent  # parent of "scripts/"
    if str(workflow_root) not in sys.path:
        sys.path.insert(0, str(workflow_root))

    print(f"[wrapsnakemake] Using workflow root: {workflow_root}")

    # Load config
    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
    else:
        config = {
            "input": parse_kv_or_list(args.input),
            "output": parse_kv_or_list(args.output),
            "params": {k: v for k, v in (p.split("=", 1) for p in (args.param or []))},
            "wildcards": {k: v for k, v in (p.split("=", 1) for p in (args.wildcard or []))}
        }

    # Inject dummy snakemake object
    snakemake = make_snakemake(config)

    # Run the script as if called directly (so breakpoint() works)
    script_code = Path(args.script).read_text()
    globals_dict = {"__name__": "__main__", "snakemake": snakemake}
    exec(compile(script_code, str(args.script), "exec"), globals_dict)
