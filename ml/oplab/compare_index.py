"""Compare two reference indexes built from the same catalog on different Macs.

Vision's feature print can differ slightly between macOS versions, and the phone runs its own OS.
Build the full index on both Macs, then compare: rows are matched by printing ID, and the per-printing
cosine similarity shows how close the two machines' embeddings are (≥ 0.99 everywhere: interchangeable).
"""

import argparse
from pathlib import Path

import numpy as np

from . import cardvision, io


def _load(index: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = io.read_json(cardvision.meta_path(index))
    vectors = np.fromfile(index, dtype="<f4").reshape(len(meta["rows"]), meta["dimension"])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    first: dict[str, np.ndarray] = {}
    for printing, vector in zip(meta["rows"], vectors):
        first.setdefault(printing, vector)  # a printing's first row is its reference art
    return meta, first


def compare(index_a: Path, index_b: Path) -> dict:
    meta_a, a = _load(index_a)
    meta_b, b = _load(index_b)
    if meta_a["backend"] != meta_b["backend"]:
        raise ValueError(f"different backends: {meta_a['backend']} vs {meta_b['backend']}")
    shared = sorted(a.keys() & b.keys())
    if not shared:
        raise ValueError("the indexes share no printings")
    cosines = np.array([float(a[p] @ b[p]) for p in shared])
    worst = int(cosines.argmin())
    return {"shared": len(shared), "only_a": len(a.keys() - b.keys()), "only_b": len(b.keys() - a.keys()),
            "min": round(float(cosines.min()), 4), "mean": round(float(cosines.mean()), 4),
            "below_0.99": int((cosines < 0.99).sum()), "worst": shared[worst]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index_a", type=Path)
    parser.add_argument("index_b", type=Path)
    args = parser.parse_args(argv)
    try:
        stats = compare(args.index_a, args.index_b)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
    print(f"{stats['shared']} shared printings (only in A: {stats['only_a']}, only in B: {stats['only_b']})")
    print(f"cosine: mean {stats['mean']}, min {stats['min']} ({stats['worst']}), {stats['below_0.99']} below 0.99")
    print("interchangeable" if stats["min"] >= 0.99 else
          "not interchangeable: build the shipped index on the Mac whose OS matches the phone best (the MacBook)")
