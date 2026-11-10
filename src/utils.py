"""Config loading, logging, seeding and run metadata."""
from __future__ import annotations

import copy
import datetime as _dt
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None, smoke_test: bool = False) -> dict[str, Any]:
    """Load the YAML config; with smoke_test=True apply the `smoke_test` overrides."""
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["smoke"] = bool(smoke_test)
    if smoke_test:
        s = cfg["smoke_test"]
        cfg["dataset"]["train_size"] = s["train_size"]
        cfg["ivf"]["nlist"] = s["nlist"]
        cfg["search"]["repetitions"] = s["repetitions"]
        # keep smoke artefacts apart from the real experiment: indexes/smoke, results/smoke/...
        cfg["paths"] = {k: (v if k == "logs" else
                            f"{v}/smoke" if k == "indexes" else
                            str(Path(v).parent / "smoke" / Path(v).name))
                        for k, v in cfg["paths"].items()}
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_logging(name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    log_dir = resolve(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / f"{name}_{stamp}.log")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def set_threads(n: int) -> None:
    import faiss
    faiss.omp_set_num_threads(int(n))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "processor": platform.processor(),
                            "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for key in ["machdep.cpu.brand_string", "hw.memsize", "hw.perflevel0.physicalcpu",
                    "hw.perflevel1.physicalcpu"]:
            try:
                info[key] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            except Exception:
                pass
    else:
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["model_name"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    try:
        import psutil
        info["total_ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    return info


def run_metadata(cfg: dict[str, Any], run_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import faiss
    meta = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cpu": cpu_info(),
        "faiss_threads": cfg["faiss"]["threads"],
        "git_commit": git_commit(),
        "config": copy.deepcopy(cfg),
    }
    if extra:
        meta.update(extra)
    return meta


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def add_common_args(parser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    parser.add_argument("--smoke-test", action="store_true",
                        help="small subset (or synthetic data) to verify the pipeline")
    parser.add_argument("--threads", type=int, help="override faiss.threads")
    parser.add_argument("--nprobe", type=int, nargs="+", help="override ivf.nprobe")
    parser.add_argument("--pq-m", type=int, nargs="+", help="override pq.m")
    parser.add_argument("--repetitions", type=int, help="override search.repetitions")
    parser.add_argument("--rebuild", action="store_true", help="rebuild indexes even if cached")


def config_from_args(args) -> dict[str, Any]:
    cfg = load_config(args.config, smoke_test=args.smoke_test)
    if args.threads is not None:
        cfg["faiss"]["threads"] = args.threads
    if args.nprobe:
        cfg["ivf"]["nprobe"] = sorted(args.nprobe)
    if args.pq_m:
        cfg["pq"]["m"] = sorted(args.pq_m)
    if args.repetitions is not None:
        cfg["search"]["repetitions"] = args.repetitions
    cfg["ivf"]["nprobe"] = [p for p in cfg["ivf"]["nprobe"] if p <= cfg["ivf"]["nlist"]]
    for m in cfg["pq"]["m"]:
        assert cfg["dataset"]["dim"] % m == 0, f"dim {cfg['dataset']['dim']} not divisible by M={m}"
    return cfg
