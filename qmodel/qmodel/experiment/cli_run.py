import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    """Run the experiment entrypoint in-process at current working directory."""
    # Parse args to keep the `qmodel-run conf.py ...` interface stable.
    parser = argparse.ArgumentParser(prog="qmodel-run")
    parser.add_argument("conf_path", type=str)
    ns, rest = parser.parse_known_args(argv)

    # Delegate to entry_main with the same argv shape as `python entry.py conf.py ...`.
    from qmodel.experiment.entry_main import main as entry_main

    entry_main([ns.conf_path] + rest)
