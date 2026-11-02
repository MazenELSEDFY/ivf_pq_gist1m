import numpy as np
import pytest

from src.datasets import (DatasetError, iter_batches, read_vecs, validate_groundtruth,
                          write_vecs, check_finite, exact_groundtruth, synthetic_clustered)


def test_fvecs_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((50, 12)).astype(np.float32)
    p = tmp_path / "a.fvecs"
    write_vecs(p, x)
    y = read_vecs(p, "f", expected_dim=12, expected_n=50)
    assert y.dtype == np.float32 and y.shape == (50, 12)
    np.testing.assert_array_equal(y, x)


def test_ivecs_roundtrip(tmp_path):
    g = np.arange(30, dtype=np.int32).reshape(3, 10)
    p = tmp_path / "g.ivecs"
    write_vecs(p, g)
    np.testing.assert_array_equal(read_vecs(p, "i"), g)


def test_wrong_dim_and_count(tmp_path):
    p = tmp_path / "a.fvecs"
    write_vecs(p, np.zeros((5, 8), np.float32))
    with pytest.raises(DatasetError, match="dimension 8"):
        read_vecs(p, "f", expected_dim=960)
    with pytest.raises(DatasetError, match="5 records"):
        read_vecs(p, "f", expected_n=6)


def test_truncated_file(tmp_path):
    p = tmp_path / "a.fvecs"
    write_vecs(p, np.zeros((5, 8), np.float32))
    data = p.read_bytes()
    p.write_bytes(data[:-4])  # drop last value
    with pytest.raises(DatasetError, match="not divisible"):
        read_vecs(p, "f")


def test_bad_record_header(tmp_path):
    p = tmp_path / "a.fvecs"
    write_vecs(p, np.zeros((5, 8), np.float32))
    raw = np.fromfile(p, dtype=np.int32).reshape(5, 9)
    raw[3, 0] = 7  # corrupt header of row 3 (file length still consistent)
    raw.tofile(p)
    with pytest.raises(DatasetError, match="first at row 3"):
        read_vecs(p, "f")


def test_missing_and_empty(tmp_path):
    with pytest.raises(DatasetError, match="missing"):
        read_vecs(tmp_path / "nope.fvecs", "f")
    (tmp_path / "e.fvecs").write_bytes(b"")
    with pytest.raises(DatasetError):
        read_vecs(tmp_path / "e.fvecs", "f")


def test_non_finite_detected():
    x = np.zeros((10, 4), np.float32)
    x[7, 2] = np.nan
    with pytest.raises(DatasetError, match="row 7"):
        check_finite(x, "x", batch=3)


def test_groundtruth_validation():
    gt = np.array([[0, 1, 2], [3, 4, 5]])
    validate_groundtruth(gt, n_queries=2, n_base=6, min_k=3)
    with pytest.raises(DatasetError, match="rows"):
        validate_groundtruth(gt, 3, 6, 3)
    with pytest.raises(DatasetError, match=">= 4"):
        validate_groundtruth(gt, 2, 6, 4)
    with pytest.raises(DatasetError, match="out of range"):
        validate_groundtruth(gt, 2, 5, 3)
    with pytest.raises(DatasetError, match="duplicate"):
        validate_groundtruth(np.array([[0, 1, 1]]), 1, 6, 3)


def test_iter_batches_contiguous_and_complete(tmp_path):
    x = np.random.default_rng(1).standard_normal((23, 6)).astype(np.float32)
    p = tmp_path / "a.fvecs"
    write_vecs(p, x)
    view = read_vecs(p, "f")
    assert not view.flags["C_CONTIGUOUS"]
    parts = []
    for s, chunk in iter_batches(view, 10):
        assert chunk.flags["C_CONTIGUOUS"] and chunk.dtype == np.float32
        parts.append((s, chunk))
    assert [s for s, _ in parts] == [0, 10, 20]
    np.testing.assert_array_equal(np.vstack([c for _, c in parts]), x)


def test_exact_groundtruth_matches_numpy():
    xt, xb, xq = synthetic_clustered(10, 500, 20, 16, seed=3)
    gt = exact_groundtruth(xb, xq, 5)
    d2 = ((xq[:, None, :] - xb[None, :, :]) ** 2).sum(-1)
    ref = np.argsort(d2, axis=1)[:, :5]
    # compare as sets of distances (robust to exact ties)
    np.testing.assert_allclose(np.take_along_axis(d2, gt, 1), np.take_along_axis(d2, ref, 1), rtol=1e-4)
