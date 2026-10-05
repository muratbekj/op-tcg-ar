import json

import pytest

from oplab import paths, registry


def test_registry_rejects_unsafe_names():
    for name in ["", "../v1", "v 1", "v1;rm", "$(x)"]:
        with pytest.raises(ValueError, match="version name"):
            registry.version_dir(name)
    assert registry.version_dir("v1") == paths.MODELS / "v1"


SUMMARY = {"n": 210, "top1": 0.81, "top1_ci": [0.75, 0.86], "ocr_accuracy": 0.6, "ocr_accuracy_ci": [0.53, 0.66],
           "within_group": 0.9, "within_group_ci": [0.8, 0.95], "detection": 0.95, "detection_ci": [0.91, 0.97]}
GROUPS = {"method": {"ocr+vision": {"n": 50, "recall": 0.9}, "vision-only": {"n": 120, "recall": 0.75}}}


def test_card_for_a_fine_tuned_version():
    training = {"parent": "v0", "created": "2026-10-05", "classes": 4212,
                "counts": {"printings": 4212, "scans": 156, "scan_printings": 40, "scans_skipped": 2, "scan_weight": 4},
                "epochs": 6, "synthetic_val_top1": 0.97}
    metrics = {"name": "v1", "testset": "test-v1", "backend": "coreml:CardEmbedder@v1", "summary": SUMMARY, "groups": GROUPS}
    card = registry.render_card("v1", training, metrics, {"macos": "15.7.3", "swift": "Apple Swift version 6.1.2", "torch": "2.14"})
    assert card.startswith("# Model card: v1")
    assert "Parent: v0" in card and "`coreml:CardEmbedder@v1`" in card
    assert "4212 catalog printings" in card and "156 real scans of 40 printings" in card
    assert "Test set: `test-v1`" in card
    assert "| Top-1 printing | 81.0% (75.0–86.0%) |" in card
    assert "| ocr+vision | 50 | 90.0% |" in card
    assert "macOS 15.7.3" in card and "Caveats" in card


def test_card_for_the_baseline_and_diagnostic_runs():
    metrics = {"name": "v0", "testset": None, "backend": "vision-featureprint-r2", "summary": SUMMARY, "groups": {}}
    card = registry.render_card("v0", None, metrics, {"macos": "15.7.3", "swift": "x", "torch": None})
    assert "Apple Vision feature print" in card and "no training" in card
    assert "diagnostic" in card.lower() and "not shippable" in card.lower()


def test_eval_version_builds_index_runs_eval_and_writes_card(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    calls = {}

    def fake_embed(argv):
        calls["embed"] = argv
        out = argv[argv.index("--out") + 1]
        (tmp_path / "models" / "v1").mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / "v1" / "printings.meta.json").write_text(json.dumps({"backend": "coreml:CardEmbedder@v1"}))
        open(out, "wb").close()

    def fake_eval(argv):
        calls["eval"] = argv
        run = tmp_path / "run"
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps({"summary": SUMMARY, "groups": GROUPS}))
        return run

    model = tmp_path / "models" / "v1" / "CardEmbedder.mlpackage"
    model.mkdir(parents=True)
    monkeypatch.setattr(registry.embeddings, "main", fake_embed)
    monkeypatch.setattr(registry.evaluate, "main", fake_eval)
    monkeypatch.setattr(registry.testsets, "load", lambda name: {"name": "test-v1"})
    monkeypatch.setattr(registry, "environment", lambda: {"macos": "15.7.3", "swift": "6.1.2", "torch": "2.14"})
    stored = registry.eval_version("v1")
    assert "--model" in calls["embed"] and str(model) in calls["embed"]
    assert calls["eval"][calls["eval"].index("--testset") + 1] == "test-v1"
    assert stored["testset"] == "test-v1" and stored["backend"] == "coreml:CardEmbedder@v1"
    assert (tmp_path / "models" / "v1" / "MODEL_CARD.md").read_text().startswith("# Model card: v1")
    assert json.loads((tmp_path / "models" / "v1" / "metrics.json").read_text())["summary"]["top1"] == 0.81


def test_eval_v0_uses_the_feature_print_and_diagnostic_skips_the_testset(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    calls = {}

    def fake_embed(argv):
        calls["embed"] = argv
        (tmp_path / "models" / "v0" / "printings.meta.json").write_text(json.dumps({"backend": "vision-featureprint-r2"}))

    def fake_eval(argv):
        calls["eval"] = argv
        run = tmp_path / "run0"
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps({"summary": SUMMARY, "groups": {}}))
        return run

    monkeypatch.setattr(registry.embeddings, "main", fake_embed)
    monkeypatch.setattr(registry.evaluate, "main", fake_eval)
    monkeypatch.setattr(registry, "environment", lambda: {"macos": "15", "swift": "6", "torch": None})
    stored = registry.eval_version("v0", diagnostic=True)
    assert "--model" not in calls["embed"] and "--testset" not in calls["eval"]
    assert stored["testset"] is None


def test_eval_refuses_a_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    with pytest.raises(registry.RegistryError, match="make export NAME=v2"):
        registry.eval_version("v2")


def _version(tmp_path, name, testset, with_model=True):
    d = tmp_path / "models" / name
    d.mkdir(parents=True)
    backend = f"coreml:CardEmbedder@{name}" if with_model else "vision-featureprint-r2"
    (d / "printings.f32").write_bytes(b"\0" * 8)
    (d / "printings.meta.json").write_text(json.dumps({"backend": backend, "dimension": 1, "rows": ["A", "B"]}))
    (d / "metrics.json").write_text(json.dumps({"name": name, "testset": testset, "backend": backend, "summary": SUMMARY}))
    (d / "MODEL_CARD.md").write_text(f"# Model card: {name}\n")
    if with_model:
        (d / "CardEmbedder.mlpackage").mkdir()
        (d / "CardEmbedder.mlpackage" / "Manifest.json").write_text("{}")
    return d


def test_ship_refuses_without_current_frozen_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    monkeypatch.setattr(registry.testsets, "load", lambda name: {"name": "test-v2"})
    _version(tmp_path, "v1", None)
    with pytest.raises(registry.RegistryError, match="test-v2"):
        registry.ship_version("v1", docs_models=tmp_path / "docs")
    _version(tmp_path, "v3", "test-v1")
    with pytest.raises(registry.RegistryError, match="test-v2"):
        registry.ship_version("v3", docs_models=tmp_path / "docs")
    with pytest.raises(registry.RegistryError, match="make eval NAME=v9"):
        registry.ship_version("v9", docs_models=tmp_path / "docs")


def test_ship_stages_model_index_and_card(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    monkeypatch.setattr(paths, "FULL_CATALOG", tmp_path / "catalog.json")
    (tmp_path / "catalog.json").write_text("[]")
    monkeypatch.setattr(registry.testsets, "load", lambda name: {"name": "test-v1"})
    monkeypatch.setattr(registry.dataset, "scan_records", lambda: [1, 2, 3])
    staged = {}
    monkeypatch.setattr(registry.shipped, "stage", lambda *a, **k: staged.update(args=a, kwargs=k) or {"name": a[0]})
    _version(tmp_path, "v1", "test-v1")
    registry.ship_version("v1", docs_models=tmp_path / "docs")
    assert staged["args"][0] == "v1" and staged["kwargs"]["model_version"] == "v1"
    assert staged["kwargs"]["model"].name == "CardEmbedder.mlpackage" and staged["kwargs"]["labels"] == 3
    assert (tmp_path / "docs" / "v1.md").read_text() == "# Model card: v1\n"


def test_ship_v0_has_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    monkeypatch.setattr(paths, "FULL_CATALOG", tmp_path / "catalog.json")
    (tmp_path / "catalog.json").write_text("[]")
    monkeypatch.setattr(registry.testsets, "load", lambda name: {"name": "test-v1"})
    monkeypatch.setattr(registry.dataset, "scan_records", lambda: [])
    staged = {}
    monkeypatch.setattr(registry.shipped, "stage", lambda *a, **k: staged.update(kwargs=k) or {"name": a[0]})
    _version(tmp_path, "v0", "test-v1", with_model=False)
    registry.ship_version("v0", docs_models=tmp_path / "docs")
    assert staged["kwargs"]["model"] is None and staged["kwargs"]["model_version"] is None


def test_eval_on_the_test_pool_is_preliminary_and_unshippable(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    calls = {}

    def fake_embed(argv):
        (tmp_path / "models" / "v1" / "printings.meta.json").write_text(json.dumps({"backend": "coreml:CardEmbedder@v1"}))

    def fake_eval(argv):
        calls["eval"] = argv
        run = tmp_path / "run"
        run.mkdir()
        (run / "metrics.json").write_text(json.dumps({"summary": SUMMARY, "groups": GROUPS}))
        return run

    (tmp_path / "models" / "v1" / "CardEmbedder.mlpackage").mkdir(parents=True)
    monkeypatch.setattr(registry.embeddings, "main", fake_embed)
    monkeypatch.setattr(registry.evaluate, "main", fake_eval)
    monkeypatch.setattr(registry, "environment", lambda: {"macos": "15", "swift": "6", "torch": None})
    stored = registry.eval_version("v1", scans="test")
    assert calls["eval"][calls["eval"].index("--scans") + 1] == "test" and "--testset" not in calls["eval"]
    assert stored["testset"] is None and stored["source"] == "test-pool"
    card = (tmp_path / "models" / "v1" / "MODEL_CARD.md").read_text().lower()
    assert "preliminary" in card and "not shippable" in card and "diagnostic" not in card
    monkeypatch.setattr(registry.testsets, "load", lambda name: {"name": "test-v1"})
    with pytest.raises(registry.RegistryError, match="unfrozen test pool"):
        registry.ship_version("v1", docs_models=tmp_path / "docs")
