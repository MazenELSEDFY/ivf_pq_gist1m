import numpy as np
import pytest

from src.metrics import per_query_recall, recall_at_k, recalls, latency_stats_ms, pareto_mask


GT = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8]])


def test_perfect_recall():
    assert recall_at_k(GT, GT, 4) == 1.0
    assert recall_at_k(GT, GT, 1) == 1.0


def test_order_within_topk_does_not_matter():
    ret = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])
    assert recall_at_k(ret, GT, 4) == 1.0
    # but Recall@1 compares only the first element
    assert recall_at_k(ret, GT, 1) == 0.0


def test_partial_recall_hand_computed():
    ret = np.array([[1, 9, 3, 10],     # top2 {1,9}∩{1,2}=1 ; top4: {1,3} -> 2
                    [6, 5, 11, 12]])   # top2 {6,5}∩{5,6}=2 ; top4: 2
    np.testing.assert_allclose(per_query_recall(ret, GT, 2), [0.5, 1.0])
    np.testing.assert_allclose(per_query_recall(ret, GT, 4), [0.5, 0.5])
    assert recall_at_k(ret, GT, 1) == 0.5
    assert recall_at_k(ret, GT, 4) == 0.5


def test_k_truncates_both_sides():
    # gt item 4 appears in retrieved but only at rank 4: not counted for K=2
    ret = np.array([[4, 1, 0, 0]])
    assert recall_at_k(ret, GT[:1], 2) == 0.5


def test_ids_do_not_leak_across_queries():
    ret = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])  # each row equals the *other* row's gt
    assert recall_at_k(ret, GT, 4) == 0.0


def test_sentinel_minus_one_never_matches():
    ret = np.array([[1, -1, -1, -1], [-1, -1, -1, -1]])
    gt = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_allclose(per_query_recall(ret, gt, 4), [0.25, 0.0])


def test_retrieved_wider_than_k():
    ret = np.hstack([GT, np.full((2, 96), 999)])
    assert recalls(ret, GT, [1, 4]) == {1: 1.0, 4: 1.0}


def test_shape_errors():
    with pytest.raises(ValueError):
        recall_at_k(GT[:1], GT, 1)
    with pytest.raises(ValueError):
        recall_at_k(GT, GT, 5)


def test_matches_naive_set_implementation_random():
    rng = np.random.default_rng(0)
    nq, k, n = 200, 100, 1000
    gt = np.array([rng.choice(n, k, replace=False) for _ in range(nq)])
    ret = np.array([rng.choice(n, k, replace=False) for _ in range(nq)])
    ret[rng.random(ret.shape) < 0.05] = -1
    for kk in (1, 10, 100):
        naive = np.mean([len(set(ret[i, :kk]) & set(gt[i, :kk])) / kk for i in range(nq)])
        assert recall_at_k(ret, gt, kk) == pytest.approx(naive)


def test_latency_stats():
    s = latency_stats_ms(np.arange(1, 101, dtype=float))
    assert s["p50"] == pytest.approx(50.5)
    assert s["mean"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)


def test_pareto():
    # (recall, latency, memory)
    r = [0.9, 0.8, 0.9, 0.5]
    l = [1.0, 1.0, 2.0, 0.5]
    m = [10, 10, 10, 1]
    np.testing.assert_array_equal(pareto_mask(r, l, m), [True, False, False, True])
