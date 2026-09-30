import json

import numpy as np
import pytest

from oplab import compare_index


def write(tmp_path, name, rows, vectors):
    np.asarray(vectors, dtype="<f4").tofile(tmp_path / f"{name}.f32")
    (tmp_path / f"{name}.meta.json").write_text(json.dumps(
        {"backend": "vision-featureprint-r2", "dimension": len(vectors[0]), "rows": rows}))
    return tmp_path / f"{name}.f32"


def test_aligns_rows_by_printing_and_reports_cosine(tmp_path):
    a = write(tmp_path, "a", ["P1", "P2", "P3"], [[1, 0], [0, 1], [1, 1]])
    b = write(tmp_path, "b", ["P2", "P1", "P4"], [[0, 2], [1, 0.1], [5, 5]])   # other order, scaled, P3/P4 unshared
    stats = compare_index.compare(a, b)
    assert stats["shared"] == 2 and stats["only_a"] == 1 and stats["only_b"] == 1
    assert stats["min"] == pytest.approx(1 / np.sqrt(1.01), abs=1e-4)          # P1: (1,0) vs (1,0.1)
    assert stats["worst"] == "P1"


def test_refuses_different_backends(tmp_path):
    a = write(tmp_path, "a", ["P1"], [[1, 0]])
    b = write(tmp_path, "b", ["P1"], [[1, 0]])
    meta = json.loads((tmp_path / "b.meta.json").read_text())
    meta["backend"] = "coreml:CardEmbedder@v1"
    (tmp_path / "b.meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="backend"):
        compare_index.compare(a, b)
