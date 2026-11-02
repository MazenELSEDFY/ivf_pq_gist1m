#!/usr/bin/env python
"""Re-aggregate raw results of a run and draw all figures."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src import benchmark, plotting
from src.utils import load_config, resolve, setup_logging


def latest_run(raw_dir: Path) -> str:
    metas = sorted(raw_dir.glob("*_metadata.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        raise SystemExit(f"no runs found in {raw_dir}")
    return json.loads(metas[-1].read_text())["run_id"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--run-id", help="defaults to the most recent run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke_test=args.smoke_test)
    setup_logging("plots", cfg["paths"]["logs"])
    raw = resolve(cfg["paths"]["results_raw"])
    run_id = args.run_id or latest_run(raw)
    summary = benchmark.aggregate(raw, run_id, resolve(cfg["paths"]["results_summary"]))
    plotting.make_all(summary, resolve(cfg["paths"]["figures"]) / run_id)


if __name__ == "__main__":
    main()
