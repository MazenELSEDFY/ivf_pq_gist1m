#!/usr/bin/env python
"""Run the query benchmark for all (index, nprobe) configurations, then aggregate."""
import argparse

import _bootstrap  # noqa: F401
from src import benchmark
from src.datasets import load_gist1m, load_smoke
from src.utils import (add_common_args, config_from_args, new_run_id, resolve, set_seeds,
                       set_threads, setup_logging)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--run-id", help="defaults to a timestamp")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    run_id = args.run_id or new_run_id("smoke" if cfg["smoke"] else "gist1m")
    log = setup_logging(f"benchmark_{run_id}", cfg["paths"]["logs"])
    set_seeds(cfg["seed"])
    set_threads(cfg["faiss"]["threads"])
    # full-value finite scan already done at validation/build time; skip it here
    ds = load_smoke(cfg) if cfg["smoke"] else load_gist1m(cfg, check_values=False)
    log.info("Run %s: threads=%d nprobe=%s M=%s k=%d reps=%d", run_id, cfg["faiss"]["threads"],
             cfg["ivf"]["nprobe"], cfg["pq"]["m"], cfg["search"]["k"], cfg["search"]["repetitions"])
    benchmark.run(cfg, ds, run_id, rebuild=args.rebuild)
    benchmark.aggregate(resolve(cfg["paths"]["results_raw"]), run_id,
                        resolve(cfg["paths"]["results_summary"]))
    return run_id


if __name__ == "__main__":
    main()
