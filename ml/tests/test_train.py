import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from oplab import export, paths, train


def test_export_version_is_the_run_name():
    assert export.model_version("v1") == "v1"


def test_smoke_training_writes_checkpoint_and_training_json(tmp_path, monkeypatch):
    art, models = tmp_path / "art", tmp_path / "models"
    art.mkdir()
    catalog = []
    for n, color in enumerate(["red", "blue", "green"]):
        printing = ["OP05-119", "OP05-119_p1", "OP06-118"][n]
        Image.new("RGB", (63, 88), color).save(art / f"{printing}.jpg")
        catalog.append({"printingId": printing, "cardId": printing.split("_")[0]})
    scan_dir = tmp_path / "scan1"
    scan_dir.mkdir()
    Image.new("RGB", (63, 88), "blue").save(scan_dir / "crop.jpg")
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(paths, "ART", art)
    monkeypatch.setattr(paths, "MODELS", models)
    monkeypatch.setattr(paths, "FULL_CATALOG", tmp_path / "catalog.json")
    monkeypatch.setattr(paths, "SHIPPED", tmp_path / "shipped")
    monkeypatch.setattr(train.dataset, "train_records",
                        lambda: [{"scanId": "scan1", "printingId": "OP05-119_p1", "path": str(scan_dir / "crop.jpg")}])
    train.main(["--name", "smoke", "--epochs", "1", "--views", "1", "--batch", "4", "--workers", "0",
                "--max-steps", "1", "--no-pretrained"])
    info = json.loads((models / "smoke" / "training.json").read_text())
    assert (models / "smoke" / "checkpoint.pt").exists()
    assert info["name"] == "smoke" and info["parent"] == "v0" and info["classes"] == 3
    assert info["counts"]["scans"] == 1 and info["counts"]["scan_weight"] == 4
