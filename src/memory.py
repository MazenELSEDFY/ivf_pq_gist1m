"""Theoretical and measured (serialized) index sizes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss

ID_BYTES = 8  # FAISS inverted lists store int64 ids
MIB = 1024 ** 2
GIB = 1024 ** 3


def code_bytes(index_type: str, d: int, m: int | None, nbits: int | None) -> int:
    if index_type == "ivfflat":
        return 4 * d
    assert m is not None and nbits is not None
    return (m * nbits + 7) // 8


def theoretical_bytes_per_vector(index_type: str, d: int, m: int | None = None,
                                 nbits: int | None = None, id_bytes: int = ID_BYTES) -> int:
    """IVF-Flat: 4d + 8.  IVF-PQ: M*nbits/8 + 8.  Payload only, no index structures."""
    return code_bytes(index_type, d, m, nbits) + id_bytes


def theoretical_index_bytes(index_type: str, n: int, d: int, nlist: int,
                            m: int | None = None, nbits: int | None = None) -> dict[str, int]:
    """Payload + the main fixed structures (coarse centroids, PQ codebooks)."""
    payload = n * theoretical_bytes_per_vector(index_type, d, m, nbits)
    centroids = nlist * d * 4
    codebooks = (2 ** nbits) * d * 4 if index_type == "ivfpq" else 0  # M * 2^nbits * (d/M) floats
    return {"payload_bytes": payload, "centroid_bytes": centroids, "codebook_bytes": codebooks,
            "total_bytes": payload + centroids + codebooks}


def serialized_size_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size


def precomputed_table_bytes(index: faiss.Index) -> int:
    """RAM used by IVF-PQ's residual lookup tables (nlist*M*2^nbits floats); not serialized."""
    idx = faiss.downcast_index(index)
    if isinstance(idx, faiss.IndexIVFPQ) and idx.use_precomputed_table:
        return int(idx.precomputed_table.size()) * 4
    return 0


def memory_record(index: faiss.Index, path: str | Path, index_type: str, n: int, d: int,
                  nlist: int, m: int | None, nbits: int | None) -> dict[str, Any]:
    size = serialized_size_bytes(path)
    theo = theoretical_index_bytes(index_type, n, d, nlist, m, nbits)
    return {
        "pq_code_bytes": code_bytes(index_type, d, m, nbits) if index_type == "ivfpq" else None,
        "theoretical_bytes_per_vector": theoretical_bytes_per_vector(index_type, d, m, nbits),
        "theoretical_payload_bytes": theo["payload_bytes"],
        "theoretical_total_bytes": theo["total_bytes"],
        "serialized_size_bytes": size,
        "serialized_size_mib": size / MIB,
        "serialized_size_gib": size / GIB,
        "serialized_bytes_per_vector": size / n,
        "precomputed_table_bytes": precomputed_table_bytes(index),
    }


def add_relative_columns(rows: list[dict[str, Any]], baseline_type: str = "ivfflat") -> None:
    """Fill compression_ratio_vs_ivfflat and memory_saving_percent in place."""
    base = [r["serialized_size_bytes"] for r in rows if r["index_type"] == baseline_type]
    if not base:
        return
    b = base[0]
    for r in rows:
        r["compression_ratio_vs_ivfflat"] = b / r["serialized_size_bytes"]
        r["memory_saving_percent"] = 100.0 * (1 - r["serialized_size_bytes"] / b)
