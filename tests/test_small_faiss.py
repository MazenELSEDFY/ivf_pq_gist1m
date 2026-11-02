"""End-to-end FAISS checks on tiny synthetic data (seconds, no GIST1M needed)."""
import copy

import faiss
import numpy as np
import pytest

from src.datasets import Dataset, exact_groundtruth, synthetic_clustered
from src.indexes import IncompatibleIndexError, build_or_load, set_nprobe
from src.memory import memory_record, theoretical_bytes_per_vector
from src.metrics import recall_at_k
from src.utils import load_config

D, NLIST, K = 96, 16, 20
N_TRAIN, N_BASE, N_Q = 6000, 5000, 50


@pytest.fixture(scope="module")
def ds():
    xt, xb, xq = synthetic_clustered(N_TRAIN, N_BASE, N_Q, D, seed=7, n_clusters=2000)
    return Dataset(xt=xt, xb=xb, xq=xq, gt=exact_groundtruth(xb, xq, K),
                   source="synthetic", gt_source="computed_exact", meta={"seed": 7})


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = load_config()
    c["dataset"]["dim"] = D
    c["ivf"]["nlist"] = NLIST
    c["pq"]["m"] = [8, 24]
    c["search"]["k"] = K
    c["build"]["add_batch_size"] = 1500   # forces several add() batches
    c["paths"]["indexes"] = str(tmp_path_factory.mktemp("idx"))
    faiss.omp_set_num_threads(2)
    return c


@pytest.fixture(scope="module")
def built(cfg, ds):
    return {(t, m): build_or_load(cfg, ds, t, m) for t, m in
            [("ivfflat", None), ("ivfpq", 8), ("ivfpq", 24)]}


def _search(index, xq, nprobe, k=K):
    set_nprobe(index, nprobe)
    return index.search(xq, k)[1]


def test_ntotal_and_trained(built):
    for (t, m), (index, meta) in built.items():
        assert index.is_trained and index.ntotal == N_BASE
        assert meta["loaded_from_cache"] is False
        assert meta["build_time_s"] == pytest.approx(meta["training_time_s"] + meta["add_time_s"])


def test_shared_coarse_quantizer(built):
    cents = [faiss.extract_index_ivf(i).quantizer.reconstruct_n(0, NLIST) for i, _ in built.values()]
    for c in cents[1:]:
        np.testing.assert_array_equal(c, cents[0])


def test_ivfflat_exhaustive_is_exact(built, ds):
    index = built[("ivfflat", None)][0]
    ids = _search(index, ds.xq, NLIST)
    assert recall_at_k(ids, ds.gt, K) == 1.0
    assert recall_at_k(ids, ds.gt, 1) == 1.0


def test_ivfflat_recall_monotone_in_nprobe(built, ds):
    index = built[("ivfflat", None)][0]
    r = [recall_at_k(_search(index, ds.xq, p), ds.gt, 10) for p in (1, 2, 4, 8, 16)]
    assert all(b >= a for a, b in zip(r, r[1:])), r
    assert r[0] < 1.0


def test_pq_is_lossy_and_more_bytes_helps(built, ds):
    r8 = recall_at_k(_search(built[("ivfpq", 8)][0], ds.xq, NLIST), ds.gt, 10)
    r24 = recall_at_k(_search(built[("ivfpq", 24)][0], ds.xq, NLIST), ds.gt, 10)
    assert r8 < 1.0
    assert r24 > r8


def test_ids_in_range(built, ds):
    for index, _ in built.values():
        ids = _search(index, ds.xq, 1, k=K)
        assert ((ids >= -1) & (ids < N_BASE)).all()


def test_memory_sizes(built, cfg):
    from src.indexes import index_path
    rec = {key: memory_record(i, index_path(cfg, *key), key[0], N_BASE, D, NLIST, key[1], 8)
           for key, (i, _) in built.items()}
    flat, p8, p24 = rec[("ivfflat", None)], rec[("ivfpq", 8)], rec[("ivfpq", 24)]
    assert flat["serialized_size_bytes"] > p24["serialized_size_bytes"] > p8["serialized_size_bytes"]
    assert theoretical_bytes_per_vector("ivfflat", 960) == 3848
    assert theoretical_bytes_per_vector("ivfpq", 960, 16, 8) == 24
    # serialized file must at least contain the theoretical payload
    for r in rec.values():
        assert r["serialized_size_bytes"] >= r["theoretical_payload_bytes"]
    assert p8["precomputed_table_bytes"] == NLIST * 8 * 256 * 4
    assert flat["precomputed_table_bytes"] == 0


def test_cache_reuse_and_incompatibility(built, cfg, ds):
    index, meta = build_or_load(cfg, ds, "ivfpq", 8)
    assert meta["loaded_from_cache"] is True and index.ntotal == N_BASE
    bad = copy.deepcopy(cfg)
    bad["seed"] = 999
    with pytest.raises(IncompatibleIndexError, match="seed"):
        build_or_load(bad, ds, "ivfpq", 8)
    other_ds = Dataset(xt=ds.xt, xb=ds.xb[:4000], xq=ds.xq, gt=ds.gt, source="synthetic",
                       gt_source="computed_exact", meta={"seed": 7})
    with pytest.raises(IncompatibleIndexError, match="dataset"):
        build_or_load(cfg, other_ds, "ivfflat")
