from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sangam.benchmark.capacity_search.config import CapacitySearchConfig
from sangam.benchmark.capacity_search.search_manager import SearchManager
from sangam.logger import init_logger
from sangam.process_lifecycle import install_termination_handler

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity search for sangam benchmark")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--min-search-granularity-pct", type=float)
    parser.add_argument("--max-qps-cap", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CapacitySearchConfig.from_yaml_file(args.config_path)

    manager = SearchManager(
        capacity_config=config,
        output_dir=args.output_dir,
        max_iterations_override=args.max_iterations,
        min_search_granularity_override=args.min_search_granularity_pct,
        max_qps_cap_override=args.max_qps_cap,
    )
    install_termination_handler()
    try:
        manager.run()
    except KeyboardInterrupt:
        logger.info("Capacity search interrupted; exiting.")
        sys.exit(130)
    logger.info(f"Capacity search complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
