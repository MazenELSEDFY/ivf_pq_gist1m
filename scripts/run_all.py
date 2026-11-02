#!/usr/bin/env python
"""build_indexes -> run_benchmark -> generate_plots, with the same CLI options."""
import sys

import _bootstrap  # noqa: F401
import build_indexes
import generate_plots
import run_benchmark

if __name__ == "__main__":
    argv = sys.argv[1:]
    build_indexes.main(argv)
    run_id = run_benchmark.main([a for a in argv if a != "--rebuild"])  # indexes already built
    plot_argv = ["--run-id", run_id] + (["--smoke-test"] if "--smoke-test" in argv else [])
    if "--config" in argv:
        i = argv.index("--config")
        plot_argv += argv[i:i + 2]
    generate_plots.main(plot_argv)
