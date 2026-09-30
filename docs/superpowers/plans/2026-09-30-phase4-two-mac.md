# Phase 4: Two-Mac Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Mac mini own the ML side and the MacBook own the app:
- scans drop into an SMB inbox on the mini and import with one command
- the mini checks its own setup
- a "shipped" folder is the contract for what the app should bundle
- the MacBook pulls it over rsync, and can start training on the mini over SSH

**Architecture:** Everything is Python in `ml/oplab/`, with thin `ml/scripts/` entry points and Make targets.
- `shipped.py`: the `ml/shipped/` contract (files plus `shipped.json`), validation, and staging, including `ship-baseline` for the current feature-print index as `v0`.
- `pull.py`: fetches (rsync, or local), validates, and installs into `data/cards/` and the app's `Resources/Models/`.
- `dataset.import_inbox`: imports and archives scan folders dropped anywhere in `~/oplab-inbox`.
- `doctor.py`: the mini setup checks, run against an injectable probe.
- `remote.py`: `ml/remote.env`, and the SSH/tmux command for `train-remote`.

Process spawning (rsync, ssh, `sharing -l`) is injected, so every rule is unit-tested without a second Mac.

**Tech Stack:** Python 3 under `uv` (pytest), rsync/ssh/tmux/caffeinate on macOS, Make.

**Spec:** `docs/superpowers/specs/2026-09-29-code-first-recognition-design.md`: "Delivery order" item 4 and section "3. Two-Mac workflow".

## Global Constraints

- **Division of work:**
  - MacBook: iOS dev and app builds.
  - Mac mini: datasets, training, evals, showcase generation.
- **What goes where:**
  - Git carries code, `results.csv`, `ml/testsets/*.json`, `docs/models/*.md`, and the chart SVG.
  - rsync carries `ml/shipped/` (mlpackage, index, `catalog.json`). `ml/shipped/` is gitignored.
- **Mac mini one-time setup,** checked by `make mini-doctor`:
  1. Xcode at the same version as the MacBook, and both Macs on the same macOS major version.
  2. `uv`, then `make ml-setup` with the `train` extra.
  3. File Sharing on, sharing `~/oplab-inbox`.
  4. Remote Login on, with the MacBook's SSH key authorized.
- **MacBook config:** `ml/remote.env` (gitignored) holds `MINI_HOST` and `MINI_REPO`.
- **`import-scans`** (mini) imports new scan folders from `~/oplab-inbox` (skipping known IDs) and moves the originals to `~/oplab-inbox/imported/`. Malformed folders are skipped, reported, and left in the inbox.
- **`status`** (mini) shows new labels since the last shipped model, test-pool progress, and the current shipped version.
- **`pull-model`** (MacBook) rsyncs `ml/shipped/` into the app's resources, and fails if the model version doesn't match the index `backend`.
- **`train-remote NAME=vN`** (MacBook) runs `make train` on the mini over SSH inside `tmux`.
- **`train NAME=vN`** (mini) fine-tunes under `caffeinate -i`.
- **Backends:**
  - An index built by the Vision feature print has backend `vision-featureprint-r2`.
  - One built by an exported model has backend `coreml:CardEmbedder@<version>`.
  - The app ignores an index whose backend doesn't match its embedder.
- **Commits:** no `Co-Authored-By` or any Claude attribution trailer. Stage files explicitly by path.
- **Uncommitted user changes** are in the working tree: `.gitignore` (an `ml/Op-Scans/*` hunk), `apps/ios/Info.plist`, `apps/ios/OnePieceAR.xcodeproj/project.pbxproj`. **Never stage them.** Task 1 changes `.gitignore`: stage only its own lines, non-interactively. Take `git show HEAD:.gitignore`, append the new lines to that copy, then `git hash-object -w` and `git update-index --cacheinfo`, and check with `git diff --cached .gitignore`.
- Python commands run from `ml/`: `cd ml && uv run pytest -q`. Never run `evaluate.py`, `generate_embeddings.py`, or anything that writes `ml/results/results.csv`, `ml/testsets/`, `data/cards/`, or the app's Resources, except inside pytest tmp dirs.

## Review Focus

1. **A shipped folder whose model and index disagree** (a model shipped with a feature-print index, a feature-print backend shipped with a model, or a missing index). `pull-model` must refuse *before* touching the app's files. Pinned by `test_install_refuses_invalid_shipment_without_touching_app` (Task 2).
2. **Pulling a feature-print shipment onto a MacBook that still has an older `CardEmbedder.mlpackage`.** The stale model must be removed, or the app would ignore the new index. Pinned by `test_install_removes_stale_model` (Task 2).
3. **The whole phone `Scans` folder copied into the inbox, including scans already imported.** Known scans are refreshed and archived, not duplicated, and the inbox is left clean except for malformed folders. Pinned by `test_import_inbox_handles_nested_scans_folder_and_known_scans` (Task 3).
4. **`remote.env` missing or incomplete** on the MacBook. `pull-model` and `train-remote` stop with a message naming the file and keys, not a traceback. Pinned by `test_mini_requires_host_and_repo` (Task 2).
5. **A `NAME` with spaces or shell metacharacters** passed to `train-remote`. It must be rejected, never interpolated into a remote shell. Pinned by `test_train_command_rejects_unsafe_names` (Task 5).

---

### Task 1: The shipped contract and `ship-baseline`

**Files:**
- Modify: `ml/oplab/paths.py` (add `SHIPPED`, `REMOTE_ENV`, `INBOX`)
- Create: `ml/oplab/shipped.py`
- Create: `ml/scripts/ship.py`
- Create: `ml/tests/test_shipped.py`
- Modify: `Makefile` (`ship-baseline`)
- Modify: `.gitignore` (add `ml/shipped/`, `ml/remote.env`, `ml/.pull-staging/`), staged with the non-interactive procedure from Global Constraints

**Interfaces:**
- Produces:
  - `paths.SHIPPED = ML / "shipped"`, `paths.REMOTE_ENV = ML / "remote.env"`, `paths.INBOX = Path.home() / "oplab-inbox"`
  - `shipped.FEATURE_PRINT = "vision-featureprint-r2"`, `shipped.MODEL_DIR = "CardEmbedder.mlpackage"`, `shipped.FILES = ("printings.f32", "printings.meta.json", "catalog.json")`, `shipped.INFO = "shipped.json"`
  - `shipped.read(directory: Path = paths.SHIPPED) -> dict | None`
  - `shipped.validate(directory: Path) -> list[str]`, which returns an empty list when valid
  - `shipped.stage(name: str, index: Path, meta: Path, catalog: Path, labels: int, directory: Path = paths.SHIPPED, model: Path | None = None, model_version: str | None = None, today: str | None = None) -> dict`, which raises `ValueError` when the result wouldn't validate, and leaves `directory` untouched in that case
  - `shipped.status_line(labeled_now: int, directory: Path = paths.SHIPPED) -> str`
  - The `shipped.json` shape is `{"name": str, "backend": str, "modelVersion": str | null, "shipped": "YYYY-MM-DD", "labels": int}`, where `labels` is the number of labeled scans when shipped.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_shipped.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_shipped.py`
Expected: `ImportError: cannot import name 'shipped' from 'oplab'`.

- [ ] **Step 3: Implement**

`ml/oplab/paths.py`: after `RESULTS = …`, add:
```python
SHIPPED = ML / "shipped"  # what the app should bundle; rsynced to the MacBook (gitignored)
REMOTE_ENV = ML / "remote.env"  # MacBook only: MINI_HOST, MINI_REPO (gitignored)
INBOX = Path.home() / "oplab-inbox"  # Mac mini: the SMB share the iPhone copies Scans into
```
`ml/oplab/shipped.py`:
```python
"""ml/shipped/: the one version the app should bundle, rsynced from the Mac mini to the MacBook.

Contents: printings.f32 + printings.meta.json (the reference index), catalog.json, shipped.json
({name, backend, modelVersion, shipped, labels}), and CardEmbedder.mlpackage when the backend is a
Core ML model. A feature-print shipment has no model. `validate` enforces that the index, the model,
and shipped.json agree, because the app silently ignores an index built by a different embedder.
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

from . import io, paths

FEATURE_PRINT = "vision-featureprint-r2"
MODEL_DIR = "CardEmbedder.mlpackage"
FILES = ("printings.f32", "printings.meta.json", "catalog.json")
INFO = "shipped.json"


def read(directory: Path = paths.SHIPPED) -> dict | None:
    path = directory / INFO
    return io.read_json(path) if path.exists() else None


def validate(directory: Path) -> list[str]:
    info = read(directory)
    if info is None:
        return [f"no {INFO} in {directory}"]
    problems = [f"missing {name}" for name in FILES if not (directory / name).exists()]
    meta_path = directory / "printings.meta.json"
    if meta_path.exists() and io.read_json(meta_path)["backend"] != info["backend"]:
        problems.append(f"index backend {io.read_json(meta_path)['backend']} doesn't match shipped backend {info['backend']}")
    if (directory / MODEL_DIR).exists():
        expected = f"coreml:CardEmbedder@{info.get('modelVersion')}"
        if info["backend"] != expected:
            problems.append(f"model version {info.get('modelVersion')} doesn't match backend {info['backend']}")
    elif info["backend"] != FEATURE_PRINT:
        problems.append(f"backend {info['backend']} needs {MODEL_DIR}, which isn't shipped")
    return problems


def stage(name: str, index: Path, meta: Path, catalog: Path, labels: int, directory: Path = paths.SHIPPED,
          model: Path | None = None, model_version: str | None = None, today: str | None = None) -> dict:
    """Replaces `directory` with this version, atomically: builds it next to the old one, validates,
    then swaps. A shipment that wouldn't validate raises ValueError and the old one stays."""
    staging = directory.with_name(directory.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy2(index, staging / "printings.f32")
    shutil.copy2(meta, staging / "printings.meta.json")
    shutil.copy2(catalog, staging / "catalog.json")
    if model is not None:
        shutil.copytree(model, staging / MODEL_DIR)
    info = {"name": name, "backend": io.read_json(meta)["backend"], "modelVersion": model_version,
            "shipped": today or date.today().isoformat(), "labels": labels}
    io.write_json(staging / INFO, info)
    problems = validate(staging)
    if problems:
        shutil.rmtree(staging)
        raise ValueError("; ".join(problems))
    shutil.rmtree(directory, ignore_errors=True)
    staging.rename(directory)
    return info


def status_line(labeled_now: int, directory: Path = paths.SHIPPED) -> str:
    info = read(directory)
    if info is None:
        return "shipped: none yet (make ship-baseline records v0)"
    return (f"shipped: {info['name']} ({info['backend']}, {info['shipped']}), "
            f"{labeled_now - info['labels']} new labels since")


def main(argv: list[str] | None = None) -> None:
    from . import dataset

    parser = argparse.ArgumentParser(description="Stage what the app should bundle into ml/shipped/.")
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline", help="ship the current Vision feature-print index (the v0 baseline)")
    baseline.add_argument("--name", default="v0")
    args = parser.parse_args(argv)

    if args.command == "baseline":
        if not paths.INDEX_META.exists():
            raise SystemExit("no index; run generate_embeddings.py --min-similarity 0.8 first")
        if io.read_json(paths.INDEX_META)["backend"] != FEATURE_PRINT:
            raise SystemExit(f"data/cards index isn't the feature print; rebuild it without --model to ship {args.name}")
        try:
            info = stage(args.name, paths.INDEX, paths.INDEX_META, paths.FULL_CATALOG, labels=len(dataset.scan_records()))
        except ValueError as error:
            raise SystemExit(f"not shipped: {error}")
        print(f"shipped {info['name']} ({info['backend']}) -> {paths.SHIPPED.relative_to(paths.REPO)}/; "
              f"on the MacBook: make pull-model")
```
`ml/scripts/ship.py`, in the same shape as the other entry points:
```python
"""Entry point; see oplab/shipped.py. Run from ml/ with: uv run scripts/ship.py --help"""

try:
    from oplab.shipped import main
except ModuleNotFoundError as error:
    raise SystemExit(f"{error}. Run the lab through uv so its environment is used:\n"
                     f"  cd ml && uv sync && uv run scripts/ship.py")

if __name__ == "__main__":
    main()
```
`Makefile`: add `ship-baseline` to `.PHONY`, and after `freeze-test` add:
```make
## Ship the current Vision feature-print index as the baseline (v0) into ml/shipped/.
ship-baseline:
	cd ml && uv run scripts/ship.py baseline --name $(or $(NAME),v0)
```
`.gitignore`: append these lines to HEAD's version, and stage them with the non-interactive procedure from Global Constraints:
```
# Two-Mac workflow
ml/shipped/
ml/shipped.staging/
ml/remote.env
ml/.pull-staging/
```
Also add the same lines to the working-tree `.gitignore` (after the user's lines), so the files are ignored locally too.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/paths.py ml/oplab/shipped.py ml/scripts/ship.py ml/tests/test_shipped.py Makefile
# .gitignore: staged via update-index as described; verify:
git diff --cached .gitignore     # must show only the "# Two-Mac workflow" lines, not ml/Op-Scans
git commit -m "Define the shipped folder contract and ship the feature-print baseline"
```

---

### Task 2: `pull-model` and `remote.env`

**Files:**
- Create: `ml/oplab/remote.py` (env reading, `main` with `pull-model`; Task 4 and Task 5 add subcommands)
- Create: `ml/oplab/pull.py`
- Create: `ml/scripts/remote.py`
- Create: `ml/remote.env.example`
- Create: `ml/tests/test_pull.py`, `ml/tests/test_remote.py`
- Modify: `Makefile` (`pull-model`)

**Interfaces:**
- Consumes: `shipped.validate`, `shipped.read`, `shipped.FILES`, `shipped.MODEL_DIR`, `paths.SHIPPED`, `paths.REMOTE_ENV` (Task 1), `paths.DATA_CARDS`, `paths.APP_MODELS`
- Produces:
  - `class remote.RemoteConfigError(Exception)`
  - `remote.read_env(path: Path = paths.REMOTE_ENV) -> dict[str, str]`
  - `remote.mini(path: Path = paths.REMOTE_ENV) -> tuple[str, str]`, which returns `(host, repo)`. `host == "local"` means "this Mac's own `ml/shipped`".
  - `class pull.PullError(Exception)`
  - `pull.fetch(host: str, repo: str, staging: Path, run=subprocess.run, local_shipped: Path = paths.SHIPPED) -> None`
  - `pull.install(staged: Path, data_cards: Path = paths.DATA_CARDS, app_models: Path = paths.APP_MODELS) -> list[str]`, which returns human-readable actions
  - `remote.main(argv)` with the subcommand `pull-model`

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_remote.py`:
```python
import pytest

from oplab import remote


def test_read_env_parses_values_comments_and_quotes(tmp_path):
    env = tmp_path / "remote.env"
    env.write_text('# the mac mini\nMINI_HOST = murat@mac-mini.local\n\nMINI_REPO="~/github/op-tcg-ar"\nNOISE\n')
    assert remote.read_env(env) == {"MINI_HOST": "murat@mac-mini.local", "MINI_REPO": "~/github/op-tcg-ar"}
    assert remote.read_env(tmp_path / "missing.env") == {}


def test_mini_requires_host_and_repo(tmp_path):
    env = tmp_path / "remote.env"
    with pytest.raises(remote.RemoteConfigError, match=r"set MINI_HOST and MINI_REPO in .*remote.env"):
        remote.mini(env)
    env.write_text("MINI_HOST=mac-mini.local\n")
    with pytest.raises(remote.RemoteConfigError):
        remote.mini(env)
    env.write_text("MINI_HOST=mac-mini.local\nMINI_REPO=~/github/op-tcg-ar\n")
    assert remote.mini(env) == ("mac-mini.local", "~/github/op-tcg-ar")
```
`ml/tests/test_pull.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_remote.py tests/test_pull.py`
Expected: import errors (`oplab.remote` / `oplab.pull` don't exist).

- [ ] **Step 3: Implement**

`ml/oplab/remote.py`:
```python
"""Working across the two Macs, from the MacBook: ml/remote.env names the Mac mini.

ml/remote.env (gitignored; copy ml/remote.env.example):
    MINI_HOST=murat@mac-mini.local     # ssh target; "local" = this Mac is also the ML Mac
    MINI_REPO=~/github/op-tcg-ar       # the repo's path on the mini
"""

import argparse
from pathlib import Path

from . import paths


class RemoteConfigError(Exception):
    pass


def read_env(path: Path = paths.REMOTE_ENV) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, # comments and lines without '=' are ignored; quotes stripped."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mini(path: Path = paths.REMOTE_ENV) -> tuple[str, str]:
    env = read_env(path)
    host, repo = env.get("MINI_HOST", ""), env.get("MINI_REPO", "")
    if not host or not repo:
        raise RemoteConfigError(f"set MINI_HOST and MINI_REPO in {path} (copy ml/remote.env.example)")
    return host, repo


def main(argv: list[str] | None = None) -> None:
    from . import pull

    parser = argparse.ArgumentParser(description="Two-Mac commands (see ml/README.md, 'Two Macs').")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull-model", help="MacBook: fetch ml/shipped/ from the mini and install it for the app build")
    args = parser.parse_args(argv)

    try:
        if args.command == "pull-model":
            pull.main()
    except (RemoteConfigError, pull.PullError) as error:
        raise SystemExit(str(error))
```
`ml/oplab/pull.py`:
```python
"""MacBook: fetch the Mac mini's ml/shipped/ and install it where the app build picks it up:
data/cards/{printings.f32, printings.meta.json, catalog.json} (copied into the bundle by the build
phase) and apps/ios/OnePieceAR/Resources/Models/CardEmbedder.mlpackage (removed for a feature-print
shipment, or the app would use the stale model and ignore the new index). Nothing is installed unless
the fetched shipment validates."""

import shutil
import subprocess
from pathlib import Path

from . import paths, remote, shipped


class PullError(Exception):
    pass


def fetch(host: str, repo: str, staging: Path, run=subprocess.run, local_shipped: Path = paths.SHIPPED) -> None:
    shutil.rmtree(staging, ignore_errors=True)
    if host == "local":
        if not local_shipped.exists():
            raise PullError(f"nothing shipped in {local_shipped}; run make ship-baseline (or make ship) first")
        shutil.copytree(local_shipped, staging)
        return
    staging.mkdir(parents=True)
    try:
        run(["rsync", "-a", "--delete", f"{host}:{repo.rstrip('/')}/ml/shipped/", f"{staging}/"], check=True)
    except subprocess.CalledProcessError as error:
        raise PullError(f"rsync from {host} failed ({error.returncode}); is Remote Login on and the key authorized?")


def install(staged: Path, data_cards: Path = paths.DATA_CARDS, app_models: Path = paths.APP_MODELS) -> list[str]:
    problems = shipped.validate(staged)
    if problems:
        raise PullError("not installed: " + "; ".join(problems))
    actions = []
    for name in shipped.FILES:
        shutil.copy2(staged / name, data_cards / name)
        actions.append(f"{name} -> {data_cards}")
    target = app_models / shipped.MODEL_DIR
    if target.exists():
        shutil.rmtree(target)
        actions.append(f"removed old {shipped.MODEL_DIR}")
    if (staged / shipped.MODEL_DIR).exists():
        app_models.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged / shipped.MODEL_DIR, target)
        actions.append(f"{shipped.MODEL_DIR} -> {app_models}")
    return actions


def main() -> None:
    host, repo = remote.mini()
    staging = paths.ML / ".pull-staging"
    fetch(host, repo, staging)
    actions = install(staging)
    info = shipped.read(staging)
    for action in actions:
        print(f"  {action}")
    print(f"installed {info['name']} ({info['backend']}, shipped {info['shipped']}); rebuild the app in Xcode")
```
`ml/scripts/remote.py`: the same entry-point shape as `ship.py`, importing `from oplab.remote import main`.

`ml/remote.env.example`:
```
# Copy to ml/remote.env on the MacBook (gitignored).
# ssh target of the Mac mini; use "local" if this Mac also does the ML work.
MINI_HOST=murat@mac-mini.local
# Path of this repo on the Mac mini.
MINI_REPO=~/github/op-tcg-ar
```
`Makefile`: add `pull-model` to `.PHONY`, and:
```make
## MacBook: fetch ml/shipped/ from the Mac mini (ml/remote.env) and install it for the next app build.
pull-model:
	cd ml && uv run scripts/remote.py pull-model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass. Then run `make pull-model` with no `ml/remote.env`. Expected: `set MINI_HOST and MINI_REPO in …/ml/remote.env (copy ml/remote.env.example)`, exit status non-zero, no traceback, and nothing changed (`git status data/cards apps` must be unchanged).

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/remote.py ml/oplab/pull.py ml/scripts/remote.py ml/remote.env.example ml/tests/test_remote.py ml/tests/test_pull.py Makefile
git commit -m "Pull the shipped model and index from the Mac mini for the app build"
```

---

### Task 3: The inbox import and the shipped line in `status`

**Files:**
- Modify: `ml/oplab/dataset.py` (extract `_import_folder`, add `import_inbox`, add an `import-inbox` subcommand, add the shipped line to `status`)
- Modify: `ml/tests/test_dataset.py`
- Modify: `Makefile` (`import-scans`)

**Interfaces:**
- Consumes: `paths.INBOX` (Task 1), `shipped.status_line` (Task 1), the existing `import_scans` behavior
- Produces:
  - `dataset._import_folder(folder: Path, scans_dir: Path) -> str`, one of `"new"`, `"updated"`, `"same"`, `"skipped"`
  - `dataset.import_inbox(inbox: Path = paths.INBOX, scans_dir: Path = paths.SCANS, stamp: str | None = None) -> dict`, which returns `{"new", "updated", "same": int, "skipped": list[str], "archived": str | None}`
  - `prepare_dataset.py import-inbox [--inbox PATH]`

- [ ] **Step 1: Write the failing tests** (append to `ml/tests/test_dataset.py`, reusing its `write_scan` helper)

```python
def test_import_inbox_handles_nested_scans_folder_and_known_scans(tmp_path):
    inbox, lab = tmp_path / "inbox", tmp_path / "lab"
    write_scan(inbox / "Scans", "s1", label="confirmed")               # the phone's whole Scans folder
    write_scan(inbox / "Scans", "s2", label="none")
    write_scan(inbox, "s3", label="corrected", corrected=True)         # a loose scan folder
    write_scan(inbox / "Scans", "bad", label="confirmed", crop=False)  # malformed: stays in the inbox
    (inbox / "Scans" / ".DS_Store").write_bytes(b"")
    result = dataset.import_inbox(inbox, lab, stamp="20261001-120000")
    assert result == {"new": 3, "updated": 0, "same": 0, "skipped": ["Scans/bad"],
                      "archived": str(inbox / "imported" / "20261001-120000")}
    assert sorted(p.name for p in lab.iterdir()) == ["s1", "s2", "s3"]
    assert sorted(p.name for p in (inbox / "imported" / "20261001-120000").iterdir()) == ["s1", "s2", "s3"]
    assert (inbox / "Scans" / "bad" / "scan.json").exists() and not (inbox / "s3").exists()

    # Next session: the phone's Scans folder is copied again (s1 relabeled on the phone, s2 unchanged).
    write_scan(inbox / "Scans", "s1", label="corrected", corrected=True)
    write_scan(inbox / "Scans", "s2", label="none")
    again = dataset.import_inbox(inbox, lab, stamp="20261002-090000")
    assert (again["new"], again["updated"], again["same"]) == (0, 1, 1)
    assert dataset.scan_label(json.loads((lab / "s1" / "scan.json").read_text())) == "corrected"
    assert sorted(p.name for p in inbox.iterdir()) == ["Scans", "imported"]   # Scans kept: "bad" is still there


def test_import_inbox_with_nothing_to_import(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert dataset.import_inbox(inbox, tmp_path / "lab", stamp="x") == {
        "new": 0, "updated": 0, "same": 0, "skipped": [], "archived": None}
    assert not (inbox / "imported").exists()
```
(`write_scan(root, scan_id, …)` calls `folder.mkdir(parents=True)`. The second session re-creates `s1` and `s2` under `inbox/Scans`, which still exists because `bad` kept it non-empty. If `write_scan` fails because the folder exists, change the helper to `mkdir(parents=True, exist_ok=True)`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_dataset.py`
Expected: FAIL (`AttributeError: … 'import_inbox'`).

- [ ] **Step 3: Implement**

In `dataset.py`, split the loop body of `import_scans` into a helper, with no behavior change for `import_scans`:
```python
def _import_folder(folder: Path, scans_dir: Path) -> str:
    """Imports one scan folder: "new", "updated" (scan.json refreshed), "same", or "skipped" (no
    crop.jpg, or an unreadable or incomplete scan.json; nothing is touched)."""
    record = folder / "scan.json"
    try:
        data = io.read_json(record)
        data["id"], data["finalPrintingID"]
    except (ValueError, KeyError, TypeError, OSError):
        return "skipped"
    if not (folder / "crop.jpg").exists():
        return "skipped"
    target = scans_dir / folder.name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, target)
        return "new"
    if not (target / "scan.json").exists() or record.read_bytes() != (target / "scan.json").read_bytes():
        shutil.copy2(record, target / "scan.json")
        return "updated"
    return "same"
```
Rewrite `import_scans`'s loop as `result = _import_folder(record.parent, scans_dir)`, counting `new` and `updated`, and appending `folder.name` to `skipped`. Keep its return shape `{"new", "updated", "skipped"}`, so the existing tests pass unchanged.

Add `from datetime import datetime` to the imports, and:
```python
def import_inbox(inbox: Path = paths.INBOX, scans_dir: Path = paths.SCANS, stamp: str | None = None) -> dict:
    """Mac mini: imports every scan folder copied anywhere into the SMB inbox (the phone's whole Scans
    folder, or loose scan folders), then archives each imported or already-known folder under
    inbox/imported/<stamp>/ so the inbox only ever holds what's new. Malformed folders stay put and
    are reported. Folders emptied by the move are removed (a leftover .DS_Store doesn't count)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = inbox / "imported" / stamp
    counts = {"new": 0, "updated": 0, "same": 0}
    skipped = []
    folders = sorted({p.parent for p in inbox.rglob("scan.json") if p.relative_to(inbox).parts[0] != "imported"})
    for folder in folders:
        result = _import_folder(folder, scans_dir)
        if result == "skipped":
            skipped.append(str(folder.relative_to(inbox)))
            continue
        counts[result] += 1
        target = archive / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(folder)
        else:
            shutil.move(str(folder), target)
    for directory in sorted((p for p in inbox.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if directory.relative_to(inbox).parts[0] == "imported":
            continue
        leftovers = [p for p in directory.iterdir() if p.name != ".DS_Store"]
        if not leftovers:
            shutil.rmtree(directory)
    archived = str(archive) if any(counts.values()) else None
    return {**counts, "skipped": skipped, "archived": archived}
```
In `main`:
- Add the subparser `inbox = sub.add_parser("import-inbox", help="Mac mini: import everything copied into ~/oplab-inbox and archive it")` with `inbox.add_argument("--inbox", type=Path, default=paths.INBOX)`.
- Add the branch:
```python
    elif args.command == "import-inbox":
        if not args.inbox.exists():
            raise SystemExit(f"no inbox at {args.inbox}; create it and share it over SMB (make mini-doctor)")
        result = import_inbox(args.inbox)
        print(f"imported {result['new']} new scans, refreshed {result['updated']} labels, "
              f"{result['same']} already known")
        if result["archived"]:
            print(f"  originals moved to {result['archived']}")
        if result["skipped"]:
            print(f"  left in the inbox (no crop.jpg or unreadable scan.json): {', '.join(result['skipped'][:10])}")
```
- In the `status` branch, after printing the testsets lines, add `from . import shipped` (inside `main`) and `print(shipped.status_line(len(scan_records())))`.

`Makefile`: add `import-scans` to `.PHONY`, and:
```make
## Mac mini: import scans copied into ~/oplab-inbox (SMB) and archive the originals.
import-scans:
	cd ml && uv run scripts/prepare_dataset.py import-inbox
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass. Then run `make status`. Expected: the usual lines, plus `shipped: none yet (make ship-baseline records v0)`. The user's real scans in `ml/datasets/raw/scans` are counted and never modified by this command.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/dataset.py ml/tests/test_dataset.py Makefile
git commit -m "Import scans from the Mac mini's SMB inbox and show the shipped version in status"
```

---

### Task 4: `mini-doctor`

**Files:**
- Create: `ml/oplab/doctor.py`
- Create: `ml/tests/test_doctor.py`
- Modify: `ml/oplab/remote.py` (the `doctor` subcommand)
- Modify: `Makefile` (`mini-doctor`, `ml-setup-train`)

**Interfaces:**
- Consumes: `paths.INBOX`, `paths.INDEX_META`, `remote.main` (Task 2)
- Produces:
  - `doctor.Check`, a dataclass with `name: str`, `ok: bool | None` (None = couldn't tell) and `detail: str`
  - `doctor.checks(probe) -> list[Check]`
  - `doctor.render(results: list[Check]) -> tuple[str, int]`, which returns `(text, exit_code)`, where the exit code is 1 if any `ok is False`
  - `doctor.SystemProbe`, the real probe, with the methods `macos_version()`, `xcode_version()`, `which(cmd)`, `has_module(name)`, `inbox_exists()`, `inbox_shared()`, `ssh_listening()`, `authorized_keys()` and `index_rows()`

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_doctor.py`:
```python
from oplab import doctor


class FakeProbe:
    def __init__(self, **overrides):
        self.values = {"macos_version": "26.0", "xcode_version": "Xcode 27.0", "which": {"uv", "tmux", "rsync"},
                       "modules": {"torch", "coremltools"}, "inbox_exists": True, "inbox_shared": True,
                       "ssh_listening": True, "authorized_keys": True, "index_rows": 4212, **overrides}

    def macos_version(self): return self.values["macos_version"]
    def xcode_version(self): return self.values["xcode_version"]
    def which(self, cmd): return cmd in self.values["which"]
    def has_module(self, name): return name in self.values["modules"]
    def inbox_exists(self): return self.values["inbox_exists"]
    def inbox_shared(self): return self.values["inbox_shared"]
    def ssh_listening(self): return self.values["ssh_listening"]
    def authorized_keys(self): return self.values["authorized_keys"]
    def index_rows(self): return self.values["index_rows"]


def test_all_good():
    results = doctor.checks(FakeProbe())
    assert [c.name for c in results] == ["macOS", "Xcode", "uv", "tmux", "training extras", "inbox folder",
                                         "inbox shared (SMB)", "Remote Login (SSH)", "MacBook key", "full-catalog index"]
    text, code = doctor.render(results)
    assert code == 0 and "✗" not in text
    assert "macOS 26.0" in text and "Xcode 27.0" in text


def test_missing_items_fail_with_hints():
    results = doctor.checks(FakeProbe(xcode_version=None, which={"rsync"}, modules=set(), inbox_exists=False,
                                      inbox_shared=False, ssh_listening=False, authorized_keys=False, index_rows=14))
    text, code = doctor.render(results)
    assert code == 1
    assert "✗ Xcode" in text and "✗ uv" in text and "brew install tmux" in text and "make ml-setup-train" in text
    assert "mkdir ~/oplab-inbox" in text and "File Sharing" in text and "Remote Login" in text
    assert "ssh-copy-id" in text and "generate_embeddings.py" in text


def test_unknown_sharing_state_is_not_a_failure():
    results = doctor.checks(FakeProbe(inbox_shared=None))
    text, code = doctor.render(results)
    assert code == 0 and "? inbox shared (SMB)" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_doctor.py`
Expected: `ImportError: cannot import name 'doctor'`.

- [ ] **Step 3: Implement**

`ml/oplab/doctor.py`:
```python
"""Mac mini setup check (`make mini-doctor`): prints ✓ / ✗ / ? per item with how to fix it.
The checks run against a probe so they can be tested without a second Mac."""

import importlib.util
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass

from . import io, paths

FULL_CATALOG_ROWS = 4000


@dataclass
class Check:
    name: str
    ok: bool | None  # None: couldn't tell
    detail: str


class SystemProbe:
    def macos_version(self) -> str:
        return platform.mac_ver()[0] or "unknown"

    def xcode_version(self) -> str | None:
        try:
            out = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.splitlines()[0] if out else None

    def which(self, cmd: str) -> bool:
        return shutil.which(cmd) is not None

    def has_module(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def inbox_exists(self) -> bool:
        return paths.INBOX.is_dir()

    def inbox_shared(self) -> bool | None:
        try:
            out = subprocess.run(["sharing", "-l"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        shared = {line.split(":", 1)[1].strip() for line in out.splitlines() if line.strip().startswith("path:")}
        return str(paths.INBOX) in shared

    def ssh_listening(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 22), timeout=0.5):
                return True
        except OSError:
            return False

    def authorized_keys(self) -> bool:
        keys = paths.Path.home() / ".ssh" / "authorized_keys"
        return keys.exists() and keys.read_text().strip() != ""

    def index_rows(self) -> int:
        return len(io.read_json(paths.INDEX_META)["rows"]) if paths.INDEX_META.exists() else 0


def checks(probe) -> list[Check]:
    xcode = probe.xcode_version()
    shared = probe.inbox_shared()
    rows = probe.index_rows()
    training = probe.has_module("torch") and probe.has_module("coremltools")
    return [
        Check("macOS", True, f"macOS {probe.macos_version()}: keep the MacBook on the same major version"),
        Check("Xcode", xcode is not None, xcode or "install Xcode (same version as the MacBook): evals run the Swift cardvision CLI"),
        Check("uv", probe.which("uv"), "found" if probe.which("uv") else "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"),
        Check("tmux", probe.which("tmux"), "found" if probe.which("tmux") else "brew install tmux (train-remote runs training inside it)"),
        Check("training extras", training, "torch + coremltools" if training else "make ml-setup-train"),
        Check("inbox folder", probe.inbox_exists(), str(paths.INBOX) if probe.inbox_exists() else "mkdir ~/oplab-inbox"),
        Check("inbox shared (SMB)", shared,
              {True: "shared", None: "couldn't read `sharing -l`; check System Settings → General → Sharing → File Sharing"}
              .get(shared, "System Settings → General → Sharing → File Sharing → + → ~/oplab-inbox")),
        Check("Remote Login (SSH)", probe.ssh_listening(),
              "on" if probe.ssh_listening() else "System Settings → General → Sharing → Remote Login"),
        Check("MacBook key", probe.authorized_keys(),
              "authorized" if probe.authorized_keys() else "on the MacBook: ssh-copy-id <user>@<this-mac>.local"),
        Check("full-catalog index", rows >= FULL_CATALOG_ROWS,
              f"{rows} rows" if rows >= FULL_CATALOG_ROWS else
              f"{rows} rows; run fetch_cards.py --art all, then generate_embeddings.py --min-similarity 0.8"),
    ]


def render(results: list[Check]) -> tuple[str, int]:
    marks = {True: "✓", False: "✗", None: "?"}
    lines = [f"{marks[c.ok]} {c.name}: {c.detail}" for c in results]
    failed = sum(c.ok is False for c in results)
    lines.append("all set" if not failed else f"{failed} to fix")
    return "\n".join(lines), 1 if failed else 0
```
In `SystemProbe.authorized_keys`, `paths.Path` must exist. `paths.py` already imports `Path`; if not, add `from pathlib import Path` to `doctor.py` and use `Path.home()`.

In `remote.main`, add the subparser `sub.add_parser("doctor", help="Mac mini: check the one-time setup")` and the branch:
```python
        elif args.command == "doctor":
            from . import doctor
            text, code = doctor.render(doctor.checks(doctor.SystemProbe()))
            print(text)
            raise SystemExit(code)
```
(Restructure `main`'s `if` into `if … elif …`.)

`Makefile`: add `mini-doctor ml-setup-train` to `.PHONY`, and:
```make
## Python lab with training extras (torch, coremltools): Mac mini.
ml-setup-train:
	cd ml && uv sync --extra train

## Mac mini: check the one-time two-Mac setup (Xcode, uv, SMB inbox, Remote Login, index).
mini-doctor:
	cd ml && uv run scripts/remote.py doctor
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass. Then run `make mini-doctor` on this Mac. Expected: a ✓/✗/? line per check and a final `all set` or `N to fix`, with no traceback. Failing lines here are expected: this Mac may not be the mini.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/doctor.py ml/tests/test_doctor.py ml/oplab/remote.py Makefile
git commit -m "Check the Mac mini's one-time setup with make mini-doctor"
```

---

### Task 5: `train`, `train-remote`, and the Two Macs docs

**Files:**
- Modify: `ml/oplab/remote.py` (`train_command`, the `train-remote` subcommand)
- Modify: `ml/tests/test_remote.py`
- Modify: `Makefile` (`train`, `train-remote`)
- Modify: `ml/README.md` (a new "Two Macs" section)

**Interfaces:**
- Consumes: `remote.mini()` (Task 2)
- Produces:
  - `remote.train_command(host: str, repo: str, name: str, extra: str = "") -> list[str]`, which raises `ValueError` for unsafe names
  - `remote.main` subcommand `train-remote NAME [--args "…"]`

- [ ] **Step 1: Write the failing tests** (append to `ml/tests/test_remote.py`)

```python
import shlex


def test_train_command_runs_make_train_in_tmux_over_ssh():
    cmd = remote.train_command("murat@mac-mini.local", "~/github/op-tcg-ar", "v1")
    assert cmd[:2] == ["ssh", "murat@mac-mini.local"]
    shell = shlex.split(cmd[2])
    assert shell[:2] == ["zsh", "-lc"]
    inner = shell[2]
    assert inner.startswith("tmux new-session -d -s train-v1 ")
    job = shlex.split(inner)[-1]
    assert job == "cd ~/github/op-tcg-ar && mkdir -p ml/runs && make train NAME=v1 2>&1 | tee ml/runs/train-v1.log"


def test_train_command_passes_extra_args():
    cmd = remote.train_command("mini", "~/repo", "v2", extra="--epochs 3")
    assert "make train NAME=v2 ARGS='--epochs 3'" in shlex.split(shlex.split(cmd[2])[2])[-1]


@pytest.mark.parametrize("name", ["", "v 1", "v1;rm -rf ~", "$(whoami)", "../v1", "v1'"])
def test_train_command_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="NAME"):
        remote.train_command("mini", "~/repo", name)
```
(The repo path stays unquoted inside the job on purpose, so the mini's shell expands `~`. That's safe because `MINI_REPO` comes from the user's own config file, not from input.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_remote.py`
Expected: FAIL (`AttributeError: … 'train_command'`).

- [ ] **Step 3: Implement**

In `remote.py`, add `import re` and `import shlex`, and:
```python
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def train_command(host: str, repo: str, name: str, extra: str = "") -> list[str]:
    """ssh command that starts `make train NAME=<name>` on the mini inside a detached tmux session,
    logging to ml/runs/train-<name>.log. A login shell (zsh -lc) so Homebrew's uv/tmux are on PATH."""
    if not SAFE_NAME.fullmatch(name) or ".." in name:
        raise ValueError(f"NAME must be letters, digits, '.', '_' or '-' (got {name!r})")
    args = f" ARGS={shlex.quote(extra)}" if extra else ""
    job = f"cd {repo} && mkdir -p ml/runs && make train NAME={name}{args} 2>&1 | tee ml/runs/train-{name}.log"
    tmux = f"tmux new-session -d -s train-{name} {shlex.quote(job)}"
    return ["ssh", host, f"zsh -lc {shlex.quote(tmux)}"]
```
In `remote.main`:
- Add the subparser `train = sub.add_parser("train-remote", help="MacBook: start make train NAME=… on the mini inside tmux")`, with `train.add_argument("name")` and `train.add_argument("--args", default="", help="extra train_embedding.py arguments")`.
- Add the branch:
```python
        elif args.command == "train-remote":
            import subprocess
            host, repo = mini()
            try:
                command = train_command(host, repo, args.name, args.args)
            except ValueError as error:
                raise SystemExit(str(error))
            subprocess.run(command, check=True)
            print(f"training {args.name} started on {host} in tmux session train-{args.name}")
            print(f"  watch:  ssh -t {host} tmux attach -t train-{args.name}   (detach: Ctrl-b d)")
            print(f"  log:    {repo}/ml/runs/train-{args.name}.log")
```
`Makefile`: add `train train-remote` to `.PHONY`, and:
```make
## Mac mini: fine-tune an embedder, keeping the Mac awake. NAME=v1 required; ARGS passes extra options.
train:
	@test -n "$(NAME)" || { echo "usage: make train NAME=v1 [ARGS='--epochs 10']"; exit 2; }
	cd ml && caffeinate -i uv run scripts/train_embedding.py --name $(NAME) $(ARGS)

## MacBook: start `make train NAME=…` on the Mac mini (ml/remote.env) inside tmux.
train-remote:
	@test -n "$(NAME)" || { echo "usage: make train-remote NAME=v1 [ARGS='--epochs 10']"; exit 2; }
	cd ml && uv run scripts/remote.py train-remote $(NAME) --args "$(ARGS)"
```

`ml/README.md`: add a section **"## Two Macs"** before "## Scripts":
```markdown
## Two Macs

The MacBook builds the app; the Mac mini holds the datasets and does training, evals, and the
showcase. Git carries code and small text artifacts; `ml/shipped/` (what the app bundles) travels by
rsync.

**Mac mini, once:** clone the repo, install Xcode (same version as the MacBook; same macOS major
version), `make ml-setup-train`, `mkdir ~/oplab-inbox`, then in System Settings → General → Sharing
turn on **File Sharing** (add `~/oplab-inbox`) and **Remote Login**. Run `make mini-doctor` until it
says "all set".

**MacBook, once:** `cp ml/remote.env.example ml/remote.env`, set `MINI_HOST` (e.g.
`murat@mac-mini.local`) and `MINI_REPO`, then `ssh-copy-id $MINI_HOST`.

**Each labeling session:**
1. iPhone → Files → Browse → ⋯ → Connect to Server → `smb://<mini>.local` → copy On My iPhone →
   OnePieceAR → **Scans** into `oplab-inbox`.
2. Mac mini: `make import-scans` (new scans + refreshed labels; originals move to
   `oplab-inbox/imported/`), then `make status`.

**Shipping to the app:** Mac mini `make ship-baseline` (the feature print as v0; fine-tuned models
ship with `make ship` in the next phase) → MacBook `make pull-model` → rebuild in Xcode. `pull-model`
refuses a shipment whose model and index disagree, and removes a stale model for a feature-print
shipment. Single Mac? Set `MINI_HOST=local`.

**Training from the MacBook:** `make train-remote NAME=v1` starts `make train NAME=v1` on the mini in
tmux (`caffeinate` keeps it awake); attach with `ssh -t $MINI_HOST tmux attach -t train-v1`.
```
Add rows to the Scripts table: `ship.py` = `stage ml/shipped/ (baseline)`, `remote.py` = `pull-model, doctor, train-remote`. Update the `prepare_dataset.py` row to include `import-inbox`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass. Then:
- `make train` with no `NAME` must print the usage line and exit 2.
- `make train-remote NAME="v 1"` must print the `NAME must be …` message. It fails at `remote.env` first if that file is missing; that's also acceptable, so report which message appeared.

**Do not** run a real `make train` or `train-remote`.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/remote.py ml/tests/test_remote.py Makefile ml/README.md
git commit -m "Start training on the Mac mini from the MacBook and document the two-Mac workflow"
```
