"""Trial subprocess driver for capacity search.

Loads a pickled `BenchmarkConfig` and invokes `run_benchmark`. Run as a
subprocess so the capacity-search controller can terminate the entire
benchmark process tree (including the spawned sangam server) on
timeout.
"""

from __future__ import annotations

import argparse
import pickle
import sys

from sangam.benchmark.main import run_benchmark
from sangam.process_lifecycle import install_termination_handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-pickle", required=True)
    args = parser.parse_args()

    with open(args.config_pickle, "rb") as file:
        config = pickle.load(file)

    install_termination_handler()
    try:
        run_benchmark(config)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
