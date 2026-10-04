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
