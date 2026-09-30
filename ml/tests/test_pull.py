import json

import pytest

from oplab import pull, shipped


def shipment(tmp_path, backend="vision-featureprint-r2", with_model=False, version=None):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "printings.f32").write_bytes(b"\1" * 16)
    (src / "printings.meta.json").write_text(json.dumps({"backend": backend, "dimension": 2, "rows": ["A", "B"]}))
    (src / "catalog.json").write_text('[{"printingId": "A"}]')
    model = None
    if with_model:
        model = src / "CardEmbedder.mlpackage"
        model.mkdir(exist_ok=True)
        (model / "Manifest.json").write_text("{}")
    out = tmp_path / "shipped"
    shipped.stage("vX", src / "printings.f32", src / "printings.meta.json", src / "catalog.json", labels=1,
                  directory=out, model=model, model_version=version, today="2026-10-01")
    return out


def targets(tmp_path):
    data, models = tmp_path / "data_cards", tmp_path / "app_models"
    data.mkdir()
    models.mkdir()
    return data, models


def test_install_copies_index_catalog_and_model(tmp_path):
    staged = shipment(tmp_path, backend="coreml:CardEmbedder@v1", with_model=True, version="v1")
    data, models = targets(tmp_path)
    actions = pull.install(staged, data, models)
    assert (data / "printings.f32").read_bytes() == b"\1" * 16
    assert json.loads((data / "printings.meta.json").read_text())["backend"] == "coreml:CardEmbedder@v1"
    assert (data / "catalog.json").exists() and (models / "CardEmbedder.mlpackage" / "Manifest.json").exists()
    assert any("CardEmbedder.mlpackage" in a for a in actions)


def test_install_removes_stale_model(tmp_path):
    staged = shipment(tmp_path)                                   # feature print: no model
    data, models = targets(tmp_path)
    (models / "CardEmbedder.mlpackage").mkdir()
    pull.install(staged, data, models)
    assert not (models / "CardEmbedder.mlpackage").exists()


def test_install_refuses_invalid_shipment_without_touching_app(tmp_path):
    staged = shipment(tmp_path)
    (staged / "printings.meta.json").write_text(json.dumps({"backend": "coreml:CardEmbedder@v9", "dimension": 2, "rows": []}))
    data, models = targets(tmp_path)
    (data / "printings.f32").write_bytes(b"old")
    (models / "CardEmbedder.mlpackage").mkdir()
    with pytest.raises(pull.PullError, match="doesn't match"):
        pull.install(staged, data, models)
    assert (data / "printings.f32").read_bytes() == b"old" and (models / "CardEmbedder.mlpackage").exists()


def test_fetch_uses_rsync_for_a_remote_mini(tmp_path):
    calls = []
    pull.fetch("murat@mac-mini.local", "~/github/op-tcg-ar/", tmp_path / "staging", run=lambda cmd, check: calls.append(cmd))
    assert calls == [["rsync", "-a", "--delete", "murat@mac-mini.local:~/github/op-tcg-ar/ml/shipped/", f"{tmp_path / 'staging'}/"]]


def test_fetch_local_copies_this_macs_shipment(tmp_path):
    staged = shipment(tmp_path)
    out = tmp_path / "staging"
    pull.fetch("local", "", out, run=lambda *a, **k: pytest.fail("no rsync for local"), local_shipped=staged)
    assert shipped.read(out)["name"] == "vX"


def test_install_failure_while_copying_model_leaves_everything_untouched(tmp_path, monkeypatch):
    staged = shipment(tmp_path, backend="coreml:CardEmbedder@v1", with_model=True, version="v1")
    data, models = targets(tmp_path)
    (data / "printings.f32").write_bytes(b"old")
    (models / "CardEmbedder.mlpackage").mkdir()
    monkeypatch.setattr(pull.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(pull.PullError, match="untouched"):
        pull.install(staged, data, models)
    assert (data / "printings.f32").read_bytes() == b"old" and (models / "CardEmbedder.mlpackage").exists()
    assert not list(data.glob("*.incoming")) and not list(models.glob("*.incoming"))


def test_install_replaces_an_existing_model(tmp_path):
    staged = shipment(tmp_path, backend="coreml:CardEmbedder@v2", with_model=True, version="v2")
    data, models = targets(tmp_path)
    old = models / "CardEmbedder.mlpackage"
    old.mkdir()
    (old / "stale.txt").write_text("x")
    pull.install(staged, data, models)
    assert (old / "Manifest.json").exists() and not (old / "stale.txt").exists()
    assert sorted(p.name for p in models.iterdir()) == ["CardEmbedder.mlpackage"]


def test_fetch_reports_missing_rsync_and_exit_codes(tmp_path):
    import subprocess

    def missing(cmd, check):
        raise FileNotFoundError("rsync")

    with pytest.raises(pull.PullError, match="rsync not found"):
        pull.fetch("h", "r", tmp_path / "s", run=missing)

    def code(n):
        def run(cmd, check):
            raise subprocess.CalledProcessError(n, cmd)
        return run

    with pytest.raises(pull.PullError, match="run make ship-baseline there"):
        pull.fetch("h", "r", tmp_path / "s", run=code(23))
    with pytest.raises(pull.PullError, match="ssh failed"):
        pull.fetch("h", "r", tmp_path / "s", run=code(255))
