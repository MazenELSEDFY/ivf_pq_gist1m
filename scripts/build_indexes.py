#!/usr/bin/env python
"""Build (or validate cached) IVF-Flat and IVF-PQ indexes."""
import argparse
import gc

import _bootstrap  # noqa: F401
from src.datasets import load_dataset
from src.indexes import build_or_load, configured_indexes
from src.utils import add_common_args, config_from_args, set_seeds, set_threads, setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    log = setup_logging("build_smoke" if cfg["smoke"] else "build", cfg["paths"]["logs"])
    set_seeds(cfg["seed"])
    set_threads(cfg["faiss"]["threads"])
    ds = load_dataset(cfg)
    for index_type, m in configured_indexes(cfg):
        _, meta = build_or_load(cfg, ds, index_type, m, rebuild=args.rebuild)
        log.info("%-28s train %.1f s | add %.1f s | build %.1f s | %d bytes | cached=%s",
                 meta.get("name", index_type), meta["training_time_s"], meta["add_time_s"],
                 meta["build_time_s"], meta["serialized_size_bytes"], meta["loaded_from_cache"])
        gc.collect()


if __name__ == "__main__":
    main()
