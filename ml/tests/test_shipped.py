import json

import pytest

from oplab import shipped


def make_sources(tmp_path, backend="vision-featureprint-r2"):
    src = tmp_path / "src"
    src.mkdir()
    (src / "printings.f32").write_bytes(b"\0" * 16)
    (src / "printings.meta.json").write_text(json.dumps({"backend": backend, "dimension": 2, "rows": ["A", "B"]}))
    (src / "catalog.json").write_text("[]")
    model = src / "CardEmbedder.mlpackage"
    model.mkdir()
    (model / "Manifest.json").write_text("{}")
    return src


def stage(tmp_path, src, **kwargs):
    return shipped.stage(kwargs.pop("name", "v0"), src / "printings.f32", src / "printings.meta.json",
                         src / "catalog.json", labels=kwargs.pop("labels", 26), directory=tmp_path / "shipped",
                         today="2026-10-01", **kwargs)


def test_stage_feature_print_baseline(tmp_path):
    info = stage(tmp_path, make_sources(tmp_path))
    assert info == {"name": "v0", "backend": "vision-featureprint-r2", "modelVersion": None,
                    "shipped": "2026-10-01", "labels": 26}
    out = tmp_path / "shipped"
    assert sorted(p.name for p in out.iterdir()) == ["catalog.json", "printings.f32", "printings.meta.json", "shipped.json"]
    assert shipped.read(out) == info and shipped.validate(out) == []


def test_stage_model_requires_matching_backend(tmp_path):
    src = make_sources(tmp_path, backend="coreml:CardEmbedder@v1")
    info = stage(tmp_path, src, name="v1", model=src / "CardEmbedder.mlpackage", model_version="v1")
    assert info["backend"] == "coreml:CardEmbedder@v1" and (tmp_path / "shipped" / "CardEmbedder.mlpackage").is_dir()
    with pytest.raises(ValueError, match="model version v2"):
        stage(tmp_path, src, name="v2", model=src / "CardEmbedder.mlpackage", model_version="v2")
    assert shipped.read(tmp_path / "shipped")["name"] == "v1"      # a refused stage leaves the old shipment


def test_validate_reports_each_problem(tmp_path):
    out = tmp_path / "shipped"
    assert shipped.validate(out) == [f"no shipped.json in {out}"]
    stage(tmp_path, make_sources(tmp_path))
    (out / "catalog.json").unlink()
    assert shipped.validate(out) == ["missing catalog.json"]
    info = json.loads((out / "shipped.json").read_text())
    info["backend"] = "coreml:CardEmbedder@v3"
    (out / "shipped.json").write_text(json.dumps(info))
    (out / "catalog.json").write_text("[]")
    assert shipped.validate(out) == [
        "index backend vision-featureprint-r2 doesn't match shipped backend coreml:CardEmbedder@v3",
        "backend coreml:CardEmbedder@v3 needs CardEmbedder.mlpackage, which isn't shipped"]


def test_status_line(tmp_path):
    assert shipped.status_line(26, tmp_path / "shipped") == "shipped: none yet (make ship-baseline records v0)"
    stage(tmp_path, make_sources(tmp_path), labels=20)
    assert shipped.status_line(26, tmp_path / "shipped") == (
        "shipped: v0 (vision-featureprint-r2, 2026-10-01), 6 new labels since")
