# Phase 5: Training on Real Scans and the Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a card embedder on catalog art plus the user's labeled train-side scans, with batches that put same-code printings side by side. Then evaluate any version (`v0` = the Vision feature print, `v1`… = fine-tuned) on the latest frozen real test set with a generated model card, and ship a version to the app only through an eval gate.

**Architecture:** All changes are in the Python ML lab (`ml/oplab/`).
- `trainset.py` (pure, torch-free): which training images exist (art views plus oversampled train-split scans), and the group-aware batch order.
- `train.py`: consumes it, and writes a plain-JSON `training.json` next to the checkpoint.
- `export.py`: versions the Core ML model as the run name, so its backend is `coreml:CardEmbedder@vN`.
- `registry.py`:
  - `eval_version` builds `ml/models/vN/`'s index with that model, runs the frozen-set eval, and writes `metrics.json` plus `MODEL_CARD.md`.
  - `ship_version` refuses unless vN was evaluated on the current frozen set, then stages `ml/shipped/` and copies the card to `docs/models/vN.md`.
- The Makefile wraps these: `train`, `export`, `eval`, `ship`.
- `mini-doctor` treats SMB sharing as optional, since the Mac mini now does everything, including the app build.

**Tech Stack:** Python 3 under `uv` (pytest; torch, torchvision and coremltools from the `train` extra), the Swift `cardvision` CLI (via existing wrappers), Make.

**Spec:** `docs/superpowers/specs/2026-09-29-code-first-recognition-design.md`: "Delivery order" item 5, and section 2 "Training" and "Model registry".

## Global Constraints

- **Classes:** all catalog printings (with downloaded art).
- **Views:** `synth.augment_card` on catalog art, plus the real scans from train-split printings, oversampled (weight configurable, **default 4×**).
- **Group-aware batches:** each batch includes whole same-code groups, so CosFace separates siblings such as base vs alt art.
- **Output:** MobileNetV3 embedder → Core ML `CardEmbedder.mlpackage`, backend ID **`coreml:CardEmbedder@vN`**.
- **Training scans** come only from `dataset.train_records()`: labeled, train split, and in no frozen set. A test printing's real scans are never used for training, at any time.
- **Registry:** `ml/models/vN/` holds `CardEmbedder.mlpackage`, the full-catalog index built with it, `metrics.json`, and `MODEL_CARD.md`.
  - **The model card holds:**
    - parent version and date
    - training data counts (catalog printings, real scans)
    - test-set version
    - the metrics with confidence intervals, by `method`
    - macOS and toolchain versions
    - known caveats
  - **`make ship`** copies the card to `docs/models/vN.md` (tracked) and the binaries to `ml/shipped/`.
- **`ship NAME=vN`** refuses unless vN was evaluated on the current frozen test set.
- **Version names** match `[A-Za-z0-9][A-Za-z0-9._-]*` (`remote.SAFE_NAME`). `v0` means the Vision feature print, with no model.
- **Commits:** no `Co-Authored-By` or any Claude attribution trailer. Stage files explicitly.
- **Uncommitted user changes** may be in the working tree (`.gitignore`, `apps/ios/Info.plist`, `apps/ios/OnePieceAR.xcodeproj/project.pbxproj`). Never stage them.
- **Never in tests or checks:** a real training run, `make eval`, `make ship`, `generate_embeddings.py`, or `evaluate.py`. These are slow, or write tracked/shared files. Tests use tmp dirs and monkeypatched collaborators.
- **Commands:** `cd ml && uv run pytest -q`. torch is available in this env; torch-dependent tests use `pytest.importorskip("torch")`.

## Review Focus

1. **Shipping a version whose metrics come from a diagnostic (manifest) eval or an older frozen set.** It must refuse and name the set it needs. Pinned by `test_ship_refuses_without_current_frozen_eval` (Task 5).
2. **Training with zero usable scans** (none labeled yet, or all frozen or test-side). It trains on art only and reports 0 scans; it doesn't crash. Pinned by `test_items_without_scans` (Task 2).
3. **A labeled train scan whose printing has no art** (no class). It's skipped and counted, never mislabeled. Pinned by `test_scans_without_a_class_are_skipped_and_counted` (Task 2).
4. **The group sampler when there are no multi-printing codes, or the batch is bigger than the data.** Every item still appears exactly once per epoch. Pinned by `test_group_batches_cover_every_item_once` (Task 2).
5. **A version name with a path separator or shell characters** passed to `make eval`/`ship`. It must be rejected before any path is built. Pinned by `test_registry_rejects_unsafe_names` (Task 4).

---

### Task 1: SMB sharing optional in `mini-doctor`

The Mac mini now builds the app too, and scans can arrive by AirDrop or cable. So the two SMB-share checks shouldn't fail the doctor.

**Files:**
- Modify: `ml/oplab/doctor.py` (the "inbox shared (SMB)" and "repo shared (SMB)" checks)
- Modify: `ml/tests/test_doctor.py`

**Interfaces:**
- Produces: in both share checks, the "not shared" state renders as `?` (ok `None`) with a hint ending "(optional: AirDrop or cable also work)" or "(optional: only needed to pull from another Mac)". A share that is listed while File Sharing is off (port 445 closed) is still ✗, because that's a broken setup.

- [ ] **Step 1: Write the failing test** (append to `ml/tests/test_doctor.py`; update `test_missing_items_fail_with_hints` so it no longer expects `✗ repo shared (SMB)`, and instead expects `? repo shared (SMB)` and `? inbox shared (SMB)`)

```python
def test_sharing_is_optional_for_a_single_mac():
    text, code = doctor.render(doctor.checks(FakeProbe(inbox_shared=False, repo_shared=False)))
    assert code == 0 and "all set" in text
    assert "? inbox shared (SMB)" in text and "AirDrop" in text
    assert "? repo shared (SMB)" in text and "another Mac" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ml && uv run pytest -q tests/test_doctor.py`
Expected: FAIL (`code == 1`).

- [ ] **Step 3: Implement.** In `checks()`, in the inbox check, use ok `None` when `shared is False`, with the detail `"not shared: System Settings → General → Sharing → File Sharing → + → ~/oplab-inbox (optional: AirDrop or cable also work)"`. In the repo check, use ok `None` when `repo_shared is False`, with the detail `f"not shared (optional: only needed to pull from another Mac): File Sharing → + → {paths.REPO}"`. Keep the "listed but File Sharing off → ✗" branches unchanged.

- [ ] **Step 4: Run the tests.** Run: `cd ml && uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/doctor.py ml/tests/test_doctor.py
git commit -m "Treat SMB sharing as optional in mini-doctor"
```

---

### Task 2: Training items and group-aware batches (`trainset.py`)

**Files:**
- Create: `ml/oplab/trainset.py`
- Create: `ml/tests/test_trainset.py`

**Interfaces:**
- Consumes: `dataset.train_records()` records (keys `printingId`, `path`, `scanId`), `dataset.card_id_of(printing_id, card_ids)`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Item:
      path: str        # image file
      label: int       # class index
      is_scan: bool

  def build_items(classes: list[str], art_paths: dict[str, str], scans: list[dict], views: int, scan_weight: int) -> tuple[list[Item], dict]
      # Per epoch: `views` art items per class, plus each usable scan repeated `scan_weight` times.
      # Returns (items, counts): counts = {"printings", "art_views", "scans", "scan_printings", "scans_skipped", "scan_weight"}
  def group_batches(labels: list[int], codes: list[str], batch_size: int, rng: random.Random) -> list[list[int]]
      # `labels[i]` is item i's class; `codes[c]` is class c's card code.
      # Returns batches of item indices covering every item exactly once.
  ```

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_trainset.py`:
```python
import random
from collections import Counter

from oplab import trainset


CLASSES = ["OP05-119", "OP05-119_p1", "OP06-118", "OP01-001"]
ART = {c: f"/art/{c}.jpg" for c in CLASSES}


def scan(scan_id, printing):
    return {"scanId": scan_id, "printingId": printing, "path": f"/scans/{scan_id}/crop.jpg"}


def test_items_combine_art_views_and_oversampled_scans():
    items, counts = trainset.build_items(CLASSES, ART, [scan("s1", "OP05-119_p1"), scan("s2", "OP06-118")],
                                         views=2, scan_weight=4)
    assert Counter(i.is_scan for i in items) == {False: 8, True: 8}
    assert sum(1 for i in items if i.path == "/scans/s1/crop.jpg") == 4
    assert {i.label for i in items if i.path == "/scans/s1/crop.jpg"} == {CLASSES.index("OP05-119_p1")}
    assert counts == {"printings": 4, "art_views": 8, "scans": 2, "scan_printings": 2, "scans_skipped": 0, "scan_weight": 4}


def test_items_without_scans():
    items, counts = trainset.build_items(CLASSES, ART, [], views=3, scan_weight=4)
    assert len(items) == 12 and not any(i.is_scan for i in items)
    assert counts["scans"] == 0 and counts["scan_printings"] == 0


def test_scans_without_a_class_are_skipped_and_counted():
    items, counts = trainset.build_items(CLASSES, ART, [scan("s1", "OP09-999"), scan("s2", "OP01-001")],
                                         views=1, scan_weight=2)
    assert counts["scans"] == 1 and counts["scans_skipped"] == 1
    assert all(i.path != "/scans/s1/crop.jpg" for i in items)


def test_group_batches_cover_every_item_once():
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    codes = ["OP05-119", "OP05-119", "OP06-118", "OP01-001"]
    for batch_size in (2, 4, 64):
        batches = trainset.group_batches(labels, codes, batch_size, random.Random(0))
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(labels)))
        assert all(len(b) <= batch_size for b in batches)
    # No multi-printing codes at all: still a plain shuffled cover.
    batches = trainset.group_batches([0, 1, 2], ["A", "B", "C"], 2, random.Random(1))
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_group_batches_keep_siblings_together():
    # Two printings of OP05-119 (classes 0, 1) among many single-printing classes.
    labels = [0] * 8 + [1] * 8 + list(range(2, 42))
    codes = ["OP05-119", "OP05-119"] + [f"OP01-{n:03d}" for n in range(40)]
    batches = trainset.group_batches(labels, codes, 8, random.Random(3))
    together = sum(1 for b in batches if {labels[i] for i in b} >= {0, 1})
    assert together >= 4   # most sibling items share a batch with the other printing
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_trainset.py`
Expected: `ModuleNotFoundError: No module named 'oplab.trainset'`.

- [ ] **Step 3: Implement** `ml/oplab/trainset.py`:

```python
"""What a training epoch is made of, without torch so it can be tested anywhere.

Items: `views` augmented views of every catalog printing's art, plus each labeled train-split scan
repeated `scan_weight` times (real scans are scarce and closer to what the phone sees). Batches are
group-aware: printings that share a card code (base, alt art, manga, reprints) are placed in the same
batch, so the CosFace loss has to separate exactly the siblings the app must tell apart.
"""

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    path: str
    label: int
    is_scan: bool


def build_items(classes: list[str], art_paths: dict[str, str], scans: list[dict], views: int,
                scan_weight: int) -> tuple[list[Item], dict]:
    index = {printing: label for label, printing in enumerate(classes)}
    items = [Item(art_paths[printing], label, False) for label, printing in enumerate(classes) for _ in range(views)]
    usable = [s for s in scans if s["printingId"] in index]
    for record in usable:
        items += [Item(record["path"], index[record["printingId"]], True)] * scan_weight
    counts = {"printings": len(classes), "art_views": len(classes) * views, "scans": len(usable),
              "scan_printings": len({s["printingId"] for s in usable}), "scans_skipped": len(scans) - len(usable),
              "scan_weight": scan_weight}
    return items, counts


def group_batches(labels: list[int], codes: list[str], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Every item index exactly once. Items of a multi-printing code are dealt round-robin across its
    printings into consecutive slots, so siblings land in the same batch; single-printing items fill in."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    for indices in by_label.values():
        rng.shuffle(indices)
    by_code: dict[str, list[int]] = defaultdict(list)
    for label in by_label:
        by_code[codes[label]].append(label)

    chunks: list[list[int]] = []
    for group in by_code.values():
        if len(group) < 2:
            chunks += [[i] for label in group for i in by_label[label]]
            continue
        queues = [list(by_label[label]) for label in group]
        while any(queues):
            chunk = [q.pop() for q in queues if q]  # one item from each sibling printing
            chunks.append(chunk)
    rng.shuffle(chunks)

    batches, current = [], []
    for chunk in chunks:
        if current and len(current) + len(chunk) > batch_size:
            batches.append(current)
            current = []
        if len(chunk) > batch_size:  # a group wider than the batch: split it
            for start in range(0, len(chunk), batch_size):
                batches.append(chunk[start:start + batch_size])
            continue
        current += chunk
    if current:
        batches.append(current)
    return batches
```

- [ ] **Step 4: Run the tests.** Run: `cd ml && uv run pytest -q`. Expected: all pass.

If `test_group_batches_keep_siblings_together` fails on the `>= 4` bound, print the batches and check that each sibling chunk has both labels. The chunks guarantee pairs, so a failure means a chunk-building bug. Don't lower the bound.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/trainset.py ml/tests/test_trainset.py
git commit -m "Build training items with oversampled scans and group-aware batches"
```

---

### Task 3: Train on scans with group-aware batches; version the export

**Files:**
- Modify: `ml/oplab/train.py` (dataset, sampler, CLI flags, `training.json`)
- Modify: `ml/oplab/export.py` (model version = run name)
- Modify: `Makefile` (`train` passes `NAME`; new `export` target)
- Create: `ml/tests/test_train.py`

**Interfaces:**
- Consumes: `trainset.build_items`, `trainset.group_batches`, `trainset.Item`, `dataset.train_records()`, `dataset.catalog_card_ids()`, `dataset.card_id_of`.
- Produces:
  - `train.main(argv)` accepts `--scan-weight` (default 4) and `--no-scans`, and writes `ml/models/<name>/training.json`: `{"name", "created", "parent", "classes": int, "counts": {...build_items counts}, "epochs", "views", "batch", "lr", "dim", "device", "synthetic_val_top1"}`.
  - `--parent` (default: the name in `ml/shipped/shipped.json`, or `"v0"`).
  - `export.model_version(name) -> str` returns `name`. The exported model's `version` is the run name, so the app backend is `coreml:CardEmbedder@<name>`.
  - `make train NAME=v1 [ARGS=…]` (existing), and `make export NAME=v1` → `uv run scripts/export_coreml.py --name $(NAME)`.

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_train.py`
Expected: FAIL (`export` has no `model_version`; `--no-pretrained` / `training.json` don't exist).

- [ ] **Step 3: Implement**

`export.py`:
- Add `def model_version(name: str) -> str: return name` with a docstring: "The Core ML model's version string; the app's backend ID becomes `coreml:CardEmbedder@<version>`, which `ml/shipped/shipped.json` and index metadata must match."
- Replace `version = f"{args.name}-{datetime.now().strftime('%Y%m%d%H%M')}"` with `version = model_version(args.name)`.
- Drop the now-unused `datetime` import if nothing else uses it.

`train.py`:
1. Add `from . import dataset, trainset` (next to the existing imports) and `import json` and `from datetime import date`.
2. Replace `CardViews` with:
```python
class TrainingViews(Dataset):
    """One augmented view per item (art or real scan); images are downscaled once and cached."""

    def __init__(self, items: list[trainset.Item]):
        self.items = items
        self.cache: dict[str, Image.Image] = {}

    def image(self, path: str) -> Image.Image:
        if path not in self.cache:
            self.cache[path] = Image.open(path).convert("RGB").resize((315, 440), Image.Resampling.BILINEAR)
        return self.cache[path]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        rng = np.random.default_rng(random.getrandbits(64))
        return to_tensor(synth.augment_card(self.image(item.path), rng)), item.label


class GroupBatchSampler(torch.utils.data.Sampler):
    """Fresh group-aware batches every epoch (trainset.group_batches)."""

    def __init__(self, labels: list[int], codes: list[str], batch_size: int, seed: int = 0):
        self.labels, self.codes, self.batch_size = labels, codes, batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        return iter(trainset.group_batches(self.labels, self.codes, self.batch_size, self.rng))

    def __len__(self) -> int:
        return len(trainset.group_batches(self.labels, self.codes, self.batch_size, random.Random(0)))
```
3. `validate`: keep it, but call it with the clean art of each class: `[data.image(art_paths[c]) for c in classes]`.
4. In `main`:
   - Add the arguments `--scan-weight` (int, default 4, help "times each labeled train scan appears per epoch"), `--no-scans` (store_true), `--parent` (default None), `--no-pretrained` (store_true, for tests).
   - After computing `classes`, build:
```python
    art_paths = {c: str(paths.ART / f"{c}.jpg") for c in classes}
    scans = [] if args.no_scans else dataset.train_records()
    items, counts = trainset.build_items(classes, art_paths, scans, args.views, args.scan_weight)
    card_ids = dataset.catalog_card_ids()
    codes = [dataset.card_id_of(c, card_ids) for c in classes]
    data = TrainingViews(items)
    loader = DataLoader(data, batch_sampler=GroupBatchSampler([i.label for i in items], codes, args.batch),
                        num_workers=args.workers, persistent_workers=args.workers > 0)
```
   - Create the model with `Embedder(args.dim, pretrained=not args.no_pretrained)`.
   - Print `f"training on {counts['printings']} printings + {counts['scans']} real scans ×{args.scan_weight} ({counts['scans_skipped']} scans without art skipped), {len(items)} items/epoch, device {dev}"`.
   - After the checkpoint is saved, write `training.json`:
```python
    parent = args.parent or ((shipped.read() or {}).get("name") or "v0")
    io.write_json(out_dir / "training.json", {
        "name": args.name, "created": date.today().isoformat(), "parent": parent, "classes": len(classes),
        "counts": counts, "epochs": args.epochs, "views": args.views, "batch": args.batch, "lr": args.lr,
        "dim": args.dim, "device": str(dev), "synthetic_val_top1": round(accuracy, 4)})
```
     Import `shipped` (`from . import shipped`). `accuracy` is the last epoch's validation value; initialize `accuracy = 0.0` before the loop.
   - Validate the run name: `if not remote.SAFE_NAME.fullmatch(args.name): raise SystemExit("--name must be letters, digits, '.', '_' or '-'")`, with `from . import remote`.

`Makefile`: add `export` to `.PHONY`, and:
```make
## Mac mini: export a trained run to Core ML (ml/models/NAME/CardEmbedder.mlpackage).
export:
	@test -n "$(NAME)" || { echo "usage: make export NAME=v1"; exit 2; }
	cd ml && uv run scripts/export_coreml.py --name $(NAME)
```

- [ ] **Step 4: Run the tests.** Run: `cd ml && uv run pytest -q`. Expected: all pass (the smoke test runs one step on CPU/MPS in a few seconds).

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/train.py ml/oplab/export.py ml/tests/test_train.py Makefile
git commit -m "Train on oversampled real scans with group-aware batches; version exports by run name"
```

---

### Task 4: `make eval NAME=vN`: index, frozen-set eval, metrics, model card

**Files:**
- Create: `ml/oplab/registry.py`
- Create: `ml/scripts/registry.py`
- Modify: `ml/oplab/evaluate.py` (`main` returns the run directory; `metrics.json` includes `"testset"`)
- Modify: `Makefile` (`eval` target)
- Create: `ml/tests/test_registry.py`

**Interfaces:**
- Consumes:
  - `evaluate.main(argv) -> Path` (changed to return `run_dir`)
  - `embeddings.main(argv)`
  - `testsets.load("latest")`
  - `remote.SAFE_NAME`
  - `doctor.SystemProbe().swift_version()`, `platform.mac_ver()`
- Produces:
  - `registry.version_dir(name: str) -> Path` (raises `ValueError` on unsafe names)
  - `registry.eval_version(name: str, diagnostic: bool = False) -> dict`, which returns the stored metrics. It writes `ml/models/<name>/{printings.f32, printings.meta.json, metrics.json, MODEL_CARD.md}`.
  - `registry.render_card(name: str, training: dict | None, metrics: dict, environment: dict) -> str`
  - The stored `metrics.json` shape is `{"name", "testset": "test-vN" | null, "run": "<run dir>", "backend", "summary": {...}, "groups": {...}}`.
  - `make eval NAME=v1` (frozen set) and `make eval NAME=v1 DIAG=1` (synthetic/photo manifest; not shippable)

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_registry.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_registry.py`
Expected: `ImportError` (no `oplab.registry`).

- [ ] **Step 3: Implement**

`evaluate.py`:
- Make `main` end with `return run_dir` (after the prints).
- In the `io.write_json(run_dir / "metrics.json", …)` call, add `"testset": testset_name or None` to the dict, alongside the existing keys.

`ml/oplab/registry.py`:
```python
"""Model versions under ml/models/<name>/ (v0 = Apple's Vision feature print, v1… = fine-tuned).

`eval_version` builds the full-catalog index with that version's embedder, scores it on the latest
frozen real test set (or the synthetic/photo manifest with diagnostic=True), and writes metrics.json
and MODEL_CARD.md. `ship_version` (ship.py) only ships versions evaluated on the current frozen set.
"""

import argparse
import platform
from pathlib import Path

from . import doctor, embeddings, evaluate, io, paths, remote, testsets

FEATURE_PRINT_VERSION = "v0"
MODEL_DIR = "CardEmbedder.mlpackage"


class RegistryError(Exception):
    pass


def version_dir(name: str) -> Path:
    if not remote.SAFE_NAME.fullmatch(name) or ".." in name:
        raise ValueError(f"version name must be letters, digits, '.', '_' or '-' (got {name!r})")
    return paths.MODELS / name


def environment() -> dict:
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"macos": platform.mac_ver()[0] or "unknown", "swift": doctor.SystemProbe().swift_version() or "unknown",
            "torch": torch_version}


def _pct(value) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def _with_ci(value, ci) -> str:
    if value is None or not ci or ci[0] is None:
        return _pct(value)
    return f"{_pct(value)} ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"


def render_card(name: str, training: dict | None, metrics: dict, environment: dict) -> str:
    s = metrics["summary"]
    lines = [f"# Model card: {name}", "",
             f"- Backend: `{metrics['backend']}`"]
    if training is None:
        lines += ["- Model: Apple Vision feature print (revision 2), pretrained; no training (the baseline)."]
    else:
        c = training["counts"]
        lines += [f"- Parent: {training['parent']} · trained {training['created']} · {training['epochs']} epochs",
                  f"- Training data: {c['printings']} catalog printings (augmented art) + {c['scans']} real scans of "
                  f"{c['scan_printings']} printings (×{c['scan_weight']} oversampled; {c['scans_skipped']} scans without "
                  "art skipped), group-aware batches",
                  f"- Synthetic validation top-1 at the end of training: {_pct(training.get('synthetic_val_top1'))}"]
    if metrics["testset"]:
        lines += [f"- Test set: `{metrics['testset']}` (frozen real scans; printings never used for training)"]
    else:
        lines += ["- Test set: none. **Diagnostic** run on the synthetic/photo manifest: not shippable, "
                  "not a headline number."]
    lines += ["", "| Metric | Value (95% CI) |", "| --- | --- |",
              f"| Top-1 printing | {_with_ci(s.get('top1'), s.get('top1_ci'))} |",
              f"| OCR accuracy | {_with_ci(s.get('ocr_accuracy'), s.get('ocr_accuracy_ci'))} |",
              f"| Within-group top-1 | {_with_ci(s.get('within_group'), s.get('within_group_ci'))} |",
              f"| Detection | {_with_ci(s.get('detection'), s.get('detection_ci'))} |", ""]
    by_method = metrics.get("groups", {}).get("method", {})
    if by_method:
        lines += ["| Method | n | Top-1 |", "| --- | --- | --- |"]
        lines += [f"| {method} | {v['n']} | {_pct(v['recall'])} |" for method, v in sorted(by_method.items())]
        lines.append("")
    lines += ["## Environment", "",
              f"- macOS {environment['macos']} · {environment['swift']}"
              + (f" · torch {environment['torch']}" if environment.get("torch") else ""), "",
              "## Caveats", "",
              "- Real scans come from one collection and one iPhone; other cards, sleeves, and lighting may differ.",
              "- Train/test split is by printing: test printings were never photographed for training, so the score "
              "measures unseen cards, but test cards share sets, layouts, and photo conditions with training ones.",
              "- The eval runs Vision/Core ML on macOS; the phone runs iOS, whose results can differ slightly.", ""]
    return "\n".join(lines)


def eval_version(name: str, diagnostic: bool = False) -> dict:
    directory = version_dir(name)
    model = directory / MODEL_DIR
    if name != FEATURE_PRINT_VERSION and not model.exists():
        raise RegistryError(f"no {model}; train and export first: make train NAME={name} && make export NAME={name}")
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "printings.f32"

    embed_args = ["--out", str(index)]
    if name != FEATURE_PRINT_VERSION:
        embed_args += ["--model", str(model)]
    embeddings.main(embed_args)

    eval_args = ["--name", name, "--index", str(index)]
    if name != FEATURE_PRINT_VERSION:
        eval_args += ["--model", str(model)]
    testset = None
    if not diagnostic:
        testset = testsets.load("latest")["name"]
        eval_args += ["--testset", testset]
    run_dir = evaluate.main(eval_args)

    run_metrics = io.read_json(Path(run_dir) / "metrics.json")
    stored = {"name": name, "testset": testset, "run": str(run_dir),
              "backend": io.read_json(directory / "printings.meta.json")["backend"],
              "summary": run_metrics["summary"], "groups": run_metrics.get("groups", {})}
    io.write_json(directory / "metrics.json", stored)
    training = io.read_json(directory / "training.json") if (directory / "training.json").exists() else None
    (directory / "MODEL_CARD.md").write_text(render_card(name, training, stored, environment()))
    return stored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="evaluate a version on the latest frozen test set and write its model card")
    ev.add_argument("name")
    ev.add_argument("--diagnostic", action="store_true", help="synthetic/photo manifest instead (not shippable)")
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            stored = eval_version(args.name, args.diagnostic)
            print(f"{args.name}: top-1 {_with_ci(stored['summary']['top1'], stored['summary'].get('top1_ci'))} "
                  f"on {stored['testset'] or 'the diagnostic manifest'}; card: {version_dir(args.name) / 'MODEL_CARD.md'}")
    except (ValueError, RegistryError, FileNotFoundError) as error:
        raise SystemExit(str(error))
```
Note: for `v0`, `embeddings.main(["--out", …])` writes the feature-print index into `ml/models/v0/`, not `data/cards/`. That's correct, because `embeddings.main` only updates `printings.json` when `out == paths.INDEX`.

`ml/scripts/registry.py`: the same entry-point shape as the other scripts, importing `from oplab.registry import main`.

`Makefile`: replace the `eval` target with:
```make
## Evaluate model version NAME (v0 = Vision feature print) on the latest frozen test set; writes
## ml/models/NAME/{metrics.json, MODEL_CARD.md}. DIAG=1 uses the synthetic/photo manifest (not shippable).
eval:
	@test -n "$(NAME)" || { echo "usage: make eval NAME=v1 [DIAG=1]"; exit 2; }
	cd ml && uv run scripts/registry.py eval $(NAME) $(if $(DIAG),--diagnostic,)
```

- [ ] **Step 4: Run the tests.** Run: `cd ml && uv run pytest -q`. Expected: all pass. `make eval` with no `NAME` prints usage and exits 2. Do **not** run a real `make eval`.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/registry.py ml/scripts/registry.py ml/oplab/evaluate.py ml/tests/test_registry.py Makefile
git commit -m "Evaluate model versions on the frozen test set and generate model cards"
```

---

### Task 5: `make ship NAME=vN` with the eval gate, and docs

**Files:**
- Modify: `ml/oplab/registry.py` (`ship_version`, `ship` subcommand)
- Modify: `ml/tests/test_registry.py`
- Modify: `Makefile` (`ship`)
- Modify: `ml/README.md` (replace the "Fine-tuning" section; update the Scripts table)

**Interfaces:**
- Consumes: `shipped.stage(...)` (Phase 4), `dataset.scan_records()`, `testsets.load("latest")`, `paths.FULL_CATALOG`, `paths.REPO`
- Produces:
  - `registry.ship_version(name: str, docs_models: Path = paths.REPO / "docs" / "models") -> dict`, which raises `RegistryError` unless `ml/models/<name>/metrics.json` exists and its `testset` equals the current latest frozen set
  - `make ship NAME=vN`

- [ ] **Step 1: Write the failing tests** (append to `ml/tests/test_registry.py`)

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_registry.py`
Expected: FAIL (`ship_version` missing).

- [ ] **Step 3: Implement**

In `registry.py`, add `dataset` and `shipped` to the `from . import …` line, plus `import shutil`, and:
```python
def ship_version(name: str, docs_models: Path = paths.REPO / "docs" / "models") -> dict:
    """Stage version `name` into ml/shipped/ for the app, only if it was evaluated on the current
    frozen test set. Copies its model card to docs/models/<name>.md (commit it)."""
    directory = version_dir(name)
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        raise RegistryError(f"{name} hasn't been evaluated: make eval NAME={name}")
    try:
        current = testsets.load("latest")["name"]
    except FileNotFoundError as error:
        raise RegistryError(f"no frozen test set yet ({error}); ship-baseline ships v0 for app testing") from error
    evaluated = io.read_json(metrics_path).get("testset")
    if evaluated != current:
        raise RegistryError(f"{name} was evaluated on {evaluated or 'the diagnostic manifest'}, not the current "
                            f"frozen set {current}: make eval NAME={name}")
    model = directory / MODEL_DIR
    has_model = model.exists()
    info = shipped.stage(name, directory / "printings.f32", directory / "printings.meta.json", paths.FULL_CATALOG,
                         labels=len(dataset.scan_records()), model=model if has_model else None,
                         model_version=name if has_model else None)
    docs_models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory / "MODEL_CARD.md", docs_models / f"{name}.md")
    return info
```
In `main`, add `sh = sub.add_parser("ship", help="ship an evaluated version to ml/shipped/ (the app's next pull-model)")` with `sh.add_argument("name")`, and the branch:
```python
        elif args.command == "ship":
            info = ship_version(args.name)
            print(f"shipped {info['name']} -> ml/shipped/; card -> docs/models/{args.name}.md (commit it); "
                  "then make pull-model and rebuild the app")
```
Catch `shipped`'s `ValueError` too; it's already in the `except` tuple.

`Makefile`: add `ship` to `.PHONY`, and:
```make
## Ship model version NAME to ml/shipped/ (refuses unless evaluated on the current frozen test set).
ship:
	@test -n "$(NAME)" || { echo "usage: make ship NAME=v1"; exit 2; }
	cd ml && uv run scripts/registry.py ship $(NAME)
```

`ml/README.md`: replace the whole "## Fine-tuning (only once feature prints plateau)" section with:
````markdown
## Training your own model (Mac mini)

```sh
make train NAME=v1                 # catalog art + your labeled train-side scans (×4), group-aware batches
make export NAME=v1                # -> ml/models/v1/CardEmbedder.mlpackage (backend coreml:CardEmbedder@v1)
make eval NAME=v0                  # once: the Vision feature print baseline on the frozen test set
make eval NAME=v1                  # builds v1's index, scores the frozen set, writes ml/models/v1/MODEL_CARD.md
make ship NAME=v1                  # only if v1 was evaluated on the current frozen set; card -> docs/models/v1.md
make pull-model                    # install into the app, then build in Xcode
```
Training reads only labeled, train-split, unfrozen scans (`dataset.train_records()`). Test printings
never train. `ARGS='--epochs 3 --views 4'` passes options through; `--scan-weight N` changes the
oversampling, `--no-scans` trains on art alone (a useful ablation). Before the first frozen test set,
`make eval NAME=v1 DIAG=1` gives a quick synthetic/photo read. It's never shippable.
Compare versions in `ml/results/results.csv`, and on the model cards' confidence intervals: a
difference smaller than the intervals is noise.
````
In the Scripts table, add `registry.py | eval and ship model versions (model cards)`, and update the `train_embedding.py` row to `fine-tune on art + labeled train scans, group-aware batches`.

- [ ] **Step 4: Run the tests.** Run: `cd ml && uv run pytest -q`. Expected: all pass. `make ship` with no `NAME` prints usage and exits 2.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/registry.py ml/tests/test_registry.py Makefile ml/README.md
git commit -m "Ship model versions only after a frozen-set eval, with model cards"
```
