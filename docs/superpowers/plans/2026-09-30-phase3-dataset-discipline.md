# Phase 3: Dataset Discipline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real phone scans trustworthy ML data:
- only user-labeled scans are used
- a leakage-free train/test split by printing
- frozen, versioned real-scan test sets (`test-vN`)
- 95% confidence intervals on every headline metric
- a documented path to record the `v0` baseline on the first frozen set

**Architecture:** All changes are in the Python ML lab (`ml/oplab/`).
- `dataset.py`:
  - reads each scan's `label` (with a fallback for older logs)
  - assigns every scan to train/test by a stable hash of its **printing ID**
  - refreshes labels on re-import
- New `testsets.py`: owns the frozen test sets. They are JSON files in `ml/testsets/`, tracked in git.
- `metrics.py`: adds Wilson intervals.
- `evaluate.py`: can score a frozen test set, and records the set's name and the intervals in `results.csv`.
- Synthetic photos and your own photos stay as diagnostics in the existing manifest. Real scans are scored only through frozen sets.

**Tech Stack:** Python 3 under `uv` (`pytest`), the existing `oplab` package, Make.

**Spec:** `docs/superpowers/specs/2026-09-29-code-first-recognition-design.md`: "Delivery order" item 3, and section "2. Labels, dataset, and training" → "Scan logs", "Split and frozen test sets", "Metrics".

## Global Constraints

- **Labels:** only `confirmed` and `corrected` scans are used for training or testing.
  - An unlabeled scan (`none`) is never data.
  - Remove the old "weak label" behavior, which counted untouched scans as correct.
- **Split:** `split(printingId)` is a stable hash of the printing ID, giving `train` (70%) or `test` (30%).
  - It replaces the per-scan hash.
  - A test printing's real scans are never used for training, at any time.
- **Freezing:** `freeze-test` creates `test-vN` only once the unfrozen test pool has **≥200 labeled scans across ≥30 printings**.
  - Otherwise it refuses and prints the progress.
  - It writes `ml/testsets/test-vN.json` (scan IDs, printing IDs, freeze date), and the file is tracked in git.
- **After a freeze:** new scans of test printings go to the pool for `test-v(N+1)`. They never go into training, and never into an existing frozen set.
- **Headline metrics** come from the latest frozen real test set. Synthetic images and negatives stay as secondary diagnostics.
- **Confidence intervals:** every headline metric is reported with a 95% Wilson confidence interval.
- **The `v0` baseline** is the existing Vision feature-print baseline, recorded on the first frozen test set.
- `import-scans` skips malformed folders (a missing `scan.json` or `crop.jpg`), reports them, and leaves them in place.
- Commits: no `Co-Authored-By` or any Claude attribution trailer. Stage files explicitly by path.
- Python commands run from `ml/`: `cd ml && uv run pytest -q ...`. Never run `evaluate.py` in tests. No task in this plan appends to `ml/results/results.csv`: no scans exist yet, so the `v0` row is recorded later by the user.

## Review Focus

1. **A label changed on the phone after an earlier export.** Re-importing must refresh `scan.json` for folders already imported, not skip them. Pinned by `test_import_refreshes_changed_labels` (Task 1).
2. **A malformed scan folder** (no `crop.jpg`, or unreadable/incomplete `scan.json`). Import skips and reports it, and `scan_records` skips it without crashing. Pinned by `test_import_skips_malformed_folders` and `test_scan_records_skip_broken_records` (Task 1).
3. **Freezing twice, or scans arriving after a freeze.** The second freeze uses only scans not already frozen, and `test-v1` never changes. Pinned by `test_second_freeze_uses_only_new_scans` (Task 2).
4. **A frozen scan relabeled later** (on the phone, then re-imported). The frozen set keeps the printing it stored as ground truth, and the scan never re-enters a pool. Pinned by `test_relabeled_frozen_scan_keeps_frozen_truth` (Task 2) and `test_testset_entries_use_frozen_truth` (Task 4).
5. **Evaluating a frozen set on a machine missing some crops** (for example, the Mac mini before import). The eval must refuse and name the missing scans, not silently score a smaller set. Pinned by `test_testset_entries_fail_on_missing_crops` (Task 4).

---

### Task 1: Labeled scans, printing-level split, label-refreshing import

**Files:**
- Modify: `ml/oplab/dataset.py` (module docstring, `scan_split` → `printing_split`, `scan_records`, `import_scans`, `build_test_manifest`, `main`'s `import-scans` branch)
- Modify: `ml/oplab/embeddings.py` (`roster_references`, `full_references`, `--with-scans` choices)
- Modify: `ml/tests/test_synth.py` (remove `test_scan_split_is_stable_and_roughly_30_percent`)
- Modify: `ml/tests/test_embeddings.py` (only if its existing test needs the new signature)
- Create: `ml/tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `dataset.TEST_FRACTION = 0.3`
  - `dataset.LABELED = ("confirmed", "corrected")`
  - `dataset.printing_split(printing_id: str) -> str`, which returns `"train"` or `"test"`
  - `dataset.scan_label(record: dict) -> str`, which returns `"confirmed"`, `"corrected"` or `"none"`
  - `dataset.scan_records(scans_dir: Path = paths.SCANS, labeled_only: bool = True) -> list[dict]`. Each dict has the keys `id` (`"scan:<scanId>"`), `scanId`, `path` (the crop), `printingId` (the final printing), `label` and `split`.
  - `dataset.import_scans(source: Path, scans_dir: Path = paths.SCANS) -> dict`, which returns `{"new": int, "updated": int, "skipped": list[str]}`
  - `embeddings.roster_references(with_scans: str)` and `embeddings.full_references(with_scans: str = "none")`, with `with_scans` in `{"none", "labeled", "corrected"}`

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_dataset.py`:
```python
import json

from oplab import dataset


def write_scan(root, scan_id, final="OP01-003", label=None, corrected=False, crop=True):
    folder = root / scan_id
    folder.mkdir(parents=True)
    record = {"id": scan_id, "date": "2026-09-30T10:00:00Z", "candidates": [], "spawnedPrintingID": final,
              "finalPrintingID": final, "corrected": corrected, "ocrCardID": None}
    if label is not None:
        record["label"] = label
    (folder / "scan.json").write_text(json.dumps(record))
    if crop:
        (folder / "crop.jpg").write_bytes(b"jpeg")
    return folder


def test_printing_split_is_stable_and_roughly_30_percent():
    ids = [f"OP{i // 1000:02d}-{i % 1000:03d}" for i in range(3000)]
    splits = [dataset.printing_split(i) for i in ids]
    assert splits == [dataset.printing_split(i) for i in ids]
    assert set(splits) == {"train", "test"}
    assert 0.25 < splits.count("test") / len(splits) < 0.35
    assert dataset.printing_split("OP01-003") == "test" and dataset.printing_split("OP05-119") == "train"


def test_scan_label_reads_label_and_falls_back_for_old_records():
    assert dataset.scan_label({"label": "confirmed", "corrected": False}) == "confirmed"
    assert dataset.scan_label({"label": "none", "corrected": False}) == "none"
    assert dataset.scan_label({"corrected": True}) == "corrected"   # logged before labels existed
    assert dataset.scan_label({"corrected": False}) == "none"        # untouched is NOT a weak label anymore


def test_scan_records_keep_only_labeled_scans_and_split_by_printing(tmp_path):
    write_scan(tmp_path, "s1", final="OP01-003", label="confirmed")
    write_scan(tmp_path, "s2", final="OP05-119", label="corrected", corrected=True)
    write_scan(tmp_path, "s3", final="OP01-003", label="none")
    write_scan(tmp_path, "s4", final="OP05-119", corrected=True)      # old format
    records = dataset.scan_records(tmp_path)
    assert [(r["scanId"], r["label"], r["split"]) for r in records] == [
        ("s1", "confirmed", "test"), ("s2", "corrected", "train"), ("s4", "corrected", "train")]
    assert records[0]["id"] == "scan:s1" and records[0]["printingId"] == "OP01-003"
    assert records[0]["path"].endswith("s1/crop.jpg")
    assert len(dataset.scan_records(tmp_path, labeled_only=False)) == 4


def test_scan_records_skip_broken_records(tmp_path):
    write_scan(tmp_path, "ok", label="confirmed")
    write_scan(tmp_path, "nocrop", label="confirmed", crop=False)
    (tmp_path / "badjson").mkdir()
    (tmp_path / "badjson" / "scan.json").write_text("{not json")
    (tmp_path / "badjson" / "crop.jpg").write_bytes(b"jpeg")
    (tmp_path / "nofinal").mkdir()
    (tmp_path / "nofinal" / "scan.json").write_text(json.dumps({"id": "nofinal", "label": "confirmed"}))
    (tmp_path / "nofinal" / "crop.jpg").write_bytes(b"jpeg")
    assert [r["scanId"] for r in dataset.scan_records(tmp_path)] == ["ok"]


def test_import_refreshes_changed_labels(tmp_path):
    phone, lab = tmp_path / "phone", tmp_path / "lab"
    write_scan(phone, "s1", label="none")
    assert dataset.import_scans(phone, lab) == {"new": 1, "updated": 0, "skipped": []}
    # The user confirms the scan on the phone later and exports again.
    record = json.loads((phone / "s1" / "scan.json").read_text())
    record["label"] = "confirmed"
    (phone / "s1" / "scan.json").write_text(json.dumps(record))
    assert dataset.import_scans(phone, lab) == {"new": 0, "updated": 1, "skipped": []}
    assert json.loads((lab / "s1" / "scan.json").read_text())["label"] == "confirmed"
    assert dataset.import_scans(phone, lab) == {"new": 0, "updated": 0, "skipped": []}


def test_import_skips_malformed_folders(tmp_path):
    phone, lab = tmp_path / "phone", tmp_path / "lab"
    write_scan(phone, "good", label="confirmed")
    write_scan(phone, "nocrop", label="confirmed", crop=False)
    result = dataset.import_scans(phone, lab)
    assert result == {"new": 1, "updated": 0, "skipped": ["nocrop"]}
    assert (phone / "nocrop" / "scan.json").exists()                  # left in place
    assert not (lab / "nocrop").exists()
```
(The split asserts on `OP01-003` → test and `OP05-119` → train were computed from `sha256(printingId) % 100 < 30`. If they fail, the hash isn't sha256 of the UTF-8 printing ID.)

Also delete `test_scan_split_is_stable_and_roughly_30_percent` from `ml/tests/test_synth.py`. The new split test replaces it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_dataset.py`
Expected: FAIL (`AttributeError: module 'oplab.dataset' has no attribute 'printing_split'` and similar).

- [ ] **Step 3: Implement in `dataset.py`**

Replace the module docstring's last paragraph (the one starting "Scan logs are split by a stable hash of the scan ID") with:
```
Scan logs are real data only once the user answered them on the phone: `label` is `confirmed` or
`corrected`. Unlabeled scans (`none`) are never used. Labeled scans are split by a stable hash of their
printing ID: ~30% of printings are test-only forever, so photos of the same physical card never land
on both sides. Test scans are scored through frozen test sets (`testsets.py`), never the manifest.
```
Replace `scan_split` and `scan_records` with:
```python
LABELED = ("confirmed", "corrected")


def printing_split(printing_id: str) -> str:
    """'test' for a stable ~30% of printings, else 'train'. Every real scan of a printing lands on the
    same side forever, so the test score measures cards the model never saw photographed."""
    bucket = int(hashlib.sha256(printing_id.encode()).hexdigest(), 16) % 100
    return "test" if bucket < TEST_FRACTION * 100 else "train"


def scan_label(record: dict) -> str:
    """confirmed | corrected | none. Records logged before labels existed only carry `corrected`."""
    label = record.get("label")
    if label in ("confirmed", "corrected", "none"):
        return label
    return "corrected" if record.get("corrected") else "none"


def scan_records(scans_dir: Path = paths.SCANS, labeled_only: bool = True) -> list[dict]:
    """Imported scans with a crop, in scan-ID order. By default only labeled ones; folders with an
    unreadable or incomplete scan.json are skipped."""
    records = []
    for record_path in sorted(scans_dir.glob("*/scan.json")):
        crop = record_path.parent / "crop.jpg"
        if not crop.exists():
            continue
        try:
            record = io.read_json(record_path)
            scan_id, printing_id = record["id"], record["finalPrintingID"]
        except (ValueError, KeyError, TypeError):
            continue
        label = scan_label(record)
        if labeled_only and label not in LABELED:
            continue
        records.append({
            "id": f"scan:{scan_id}",
            "scanId": scan_id,
            "path": str(crop),
            "printingId": printing_id,
            "label": label,
            "split": printing_split(printing_id),
        })
    return records
```
Replace `import_scans` with:
```python
def import_scans(source: Path, scans_dir: Path = paths.SCANS) -> dict:
    """Copies new scan folders from an exported Scans folder and refreshes scan.json of already
    imported ones (a scan can be confirmed or corrected on the phone after an earlier export).
    Folders without crop.jpg are skipped, reported, and left where they are."""
    new, updated, skipped = 0, 0, []
    for record in sorted(source.glob("*/scan.json")):
        folder = record.parent
        if not (folder / "crop.jpg").exists():
            skipped.append(folder.name)
            continue
        target = scans_dir / folder.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, target)
            new += 1
        elif record.read_bytes() != (target / "scan.json").read_bytes():
            shutil.copy2(record, target / "scan.json")
            updated += 1
    return {"new": new, "updated": updated, "skipped": skipped}
```
In `build_test_manifest`, delete the `for record in scan_records(): …` loop. Change its docstring to: `"""Diagnostics: your photos, synthetic photos, and negatives. Real scans are scored through frozen test sets (testsets.py)."""`

In `main`, replace the `import-scans` print with:
```python
        result = import_scans(args.source)
        print(f"imported {result['new']} new scans, refreshed {result['updated']} labels")
        if result["skipped"]:
            print(f"  skipped {len(result['skipped'])} folders without crop.jpg: {', '.join(result['skipped'][:10])}")
```

- [ ] **Step 4: Update `embeddings.py`**

Move scan references into one helper used by both scopes. In `roster_references`, delete the `if with_scans != "none": …` block, and after the art loop add `entries += scan_references(with_scans, {e["printingId"] for e in entries}, seen)`. Add:
```python
def scan_references(with_scans: str, printing_ids: set[str] | None, seen: set[str]) -> list[dict]:
    """Labeled device scans of train-split printings as extra reference rows (never test printings).
    `labeled`: confirmed and corrected; `corrected`: corrections only. `printing_ids` limits them to
    those printings (roster scope); None allows any. `seen` holds digests already embedded."""
    if with_scans == "none":
        return []
    entries = []
    for record in dataset.scan_records():
        if record["split"] != "train" or (with_scans == "corrected" and record["label"] != "corrected"):
            continue
        if printing_ids is not None and record["printingId"] not in printing_ids:
            continue
        digest = _digest(Path(record["path"]))
        if digest not in seen:
            seen.add(digest)
            entries.append({"printingId": record["printingId"], "path": record["path"], "source": "scan"})
    return entries
```
`roster_references` already keeps its digests in a local `seen` set; pass that set in. In `full_references`, call `roster_references("none")`, build `seen` as today, append the catalog art, then `entries += scan_references(with_scans, None, seen)`. In `main`, change `--with-scans` to `choices=["none", "labeled", "corrected"]` with help `"add labeled device scans of train-split printings as extra reference rows: labeled = confirmed+corrected, corrected = corrections only"`. Update the module docstring bullet about `--with-scans` to match.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass. If `tests/test_embeddings.py` breaks only because of the moved scan block, adjust its call. It uses `with_scans` `"none"`, so it shouldn't need to change.

- [ ] **Step 6: Commit**

```bash
git add ml/oplab/dataset.py ml/oplab/embeddings.py ml/tests/test_dataset.py ml/tests/test_synth.py ml/tests/test_embeddings.py
git commit -m "Use only labeled scans, split them by printing, and refresh labels on re-import"
```

---

### Task 2: Frozen test sets and status

**Files:**
- Create: `ml/oplab/testsets.py`
- Modify: `ml/oplab/paths.py` (add `TESTSETS`)
- Modify: `ml/oplab/dataset.py` (`main`: `freeze-test` and `status` subcommands)
- Create: `ml/testsets/.gitkeep` (empty; the directory is tracked)
- Modify: `Makefile` (repo root: `status`, `freeze-test` targets)
- Create: `ml/tests/test_testsets.py`

**Interfaces:**
- Consumes: `dataset.scan_records(scans_dir, labeled_only)` records (keys `scanId`, `printingId`, `label`, `split`, `path`) from Task 1.
- Produces:
  - `paths.TESTSETS = ML / "testsets"`
  - `testsets.MIN_SCANS = 200`, `testsets.MIN_PRINTINGS = 30`
  - `class testsets.FreezeError(Exception)`
  - `testsets.frozen_sets(directory: Path = paths.TESTSETS) -> list[dict]`, sorted by `version`
  - `testsets.pool(records: list[dict], sets: list[dict]) -> list[dict]`: labeled test-split records not in any frozen set
  - `testsets.freeze(records, directory=paths.TESTSETS, min_scans=MIN_SCANS, min_printings=MIN_PRINTINGS, today: str | None = None) -> dict`
  - `testsets.load(name: str, directory: Path = paths.TESTSETS) -> dict`: `name` is `"test-vN"` or `"latest"`, and it raises `FileNotFoundError` when missing
  - `testsets.status_lines(all_records: list[dict], sets: list[dict], min_scans=MIN_SCANS, min_printings=MIN_PRINTINGS) -> list[str]`
  - The frozen set's JSON shape is `{"name": "test-vN", "version": N, "frozen": "YYYY-MM-DD", "scans": [{"scanId", "printingId"}, …], "printings": [sorted unique printing IDs]}`

- [ ] **Step 1: Write the failing tests**

`ml/tests/test_testsets.py`:
```python
import pytest

from oplab import io, testsets


def record(scan_id, printing, split="test", label="confirmed"):
    return {"id": f"scan:{scan_id}", "scanId": scan_id, "path": f"/x/{scan_id}/crop.jpg", "printingId": printing,
            "label": label, "split": split}


def test_pool_is_labeled_test_scans_not_yet_frozen():
    records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P3")]
    sets = [{"name": "test-v1", "version": 1, "scans": [{"scanId": "a", "printingId": "P1"}], "printings": ["P1"]}]
    assert [r["scanId"] for r in testsets.pool(records, sets)] == ["c"]


def test_freeze_refuses_below_thresholds_and_reports_progress(tmp_path):
    records = [record(f"s{i}", f"P{i % 3}") for i in range(10)]
    with pytest.raises(testsets.FreezeError, match="10/200 labeled scans across 3/30 printings"):
        testsets.freeze(records, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_freeze_writes_versioned_set(tmp_path):
    records = [record(f"s{i:02d}", f"P{i % 4}") for i in range(8)] + [record("t1", "P9", split="train")]
    frozen = testsets.freeze(records, tmp_path, min_scans=8, min_printings=4, today="2026-10-01")
    assert frozen["name"] == "test-v1" and frozen["version"] == 1 and frozen["frozen"] == "2026-10-01"
    assert [s["scanId"] for s in frozen["scans"]] == [f"s{i:02d}" for i in range(8)]
    assert frozen["printings"] == ["P0", "P1", "P2", "P3"]
    assert io.read_json(tmp_path / "test-v1.json") == frozen
    assert testsets.load("latest", tmp_path) == frozen == testsets.load("test-v1", tmp_path)


def test_second_freeze_uses_only_new_scans(tmp_path):
    first = [record(f"a{i}", f"P{i % 2}") for i in range(4)]
    testsets.freeze(first, tmp_path, min_scans=4, min_printings=2, today="2026-10-01")
    before = (tmp_path / "test-v1.json").read_text()
    later = first + [record(f"b{i}", f"P{i % 3}") for i in range(3)]
    second = testsets.freeze(later, tmp_path, min_scans=3, min_printings=3, today="2026-11-01")
    assert second["name"] == "test-v2" and [s["scanId"] for s in second["scans"]] == ["b0", "b1", "b2"]
    assert (tmp_path / "test-v1.json").read_text() == before
    assert [s["name"] for s in testsets.frozen_sets(tmp_path)] == ["test-v1", "test-v2"]
    assert testsets.load("latest", tmp_path)["name"] == "test-v2"


def test_relabeled_frozen_scan_keeps_frozen_truth(tmp_path):
    testsets.freeze([record("a", "P1"), record("b", "P2")], tmp_path, min_scans=2, min_printings=2, today="2026-10-01")
    # Later the user corrects scan "a" to another test printing; it must not re-enter the pool.
    relabeled = [record("a", "P7", label="corrected"), record("b", "P2")]
    assert testsets.pool(relabeled, testsets.frozen_sets(tmp_path)) == []
    assert testsets.load("test-v1", tmp_path)["scans"][0] == {"scanId": "a", "printingId": "P1"}


def test_load_missing_set_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        testsets.load("latest", tmp_path)
    with pytest.raises(FileNotFoundError):
        testsets.load("test-v3", tmp_path)


def test_status_lines():
    all_records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P1", label="none"),
                   record("d", "P3", label="corrected")]
    sets = [{"name": "test-v1", "version": 1, "frozen": "2026-10-01", "scans": [{"scanId": "d", "printingId": "P3"}],
             "printings": ["P3"]}]
    lines = testsets.status_lines(all_records, sets, min_scans=200, min_printings=30)
    assert lines == [
        "scans: 4 imported, 3 labeled (2 confirmed, 1 corrected), 1 unlabeled",
        "labeled split: 1 train, 2 test",
        "test pool for test-v2: 1/200 scans across 1/30 printings",
        "frozen: test-v1 (2026-10-01, 1 scans, 1 printings)",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_testsets.py`
Expected: `ModuleNotFoundError: No module named 'oplab.testsets'` (or an import error).

- [ ] **Step 3: Implement**

`ml/oplab/paths.py`: after the `TEST_MANIFEST` line, add:
```python
TESTSETS = ML / "testsets"  # frozen real-scan test sets, tracked in git
```

`ml/oplab/testsets.py`:
```python
"""Frozen real-scan test sets: ml/testsets/test-vN.json, tracked in git.

The test pool is every labeled scan of a test-split printing that isn't in a frozen set yet. Once it
holds at least MIN_SCANS scans across MIN_PRINTINGS printings, `freeze` writes the next test-vN with
those exact scan IDs and their labeled printings. A frozen set never changes: its stored printing is
the ground truth even if the scan is relabeled later, and its scans never return to a pool. Every
model is compared on the same frozen set, so its history stays comparable.
"""

from datetime import date
from pathlib import Path

from . import io, paths

MIN_SCANS = 200
MIN_PRINTINGS = 30


class FreezeError(Exception):
    pass


def frozen_sets(directory: Path = paths.TESTSETS) -> list[dict]:
    sets = [io.read_json(path) for path in directory.glob("test-v*.json")]
    return sorted(sets, key=lambda s: s["version"])


def _frozen_ids(sets: list[dict]) -> set[str]:
    return {scan["scanId"] for s in sets for scan in s["scans"]}


def pool(records: list[dict], sets: list[dict]) -> list[dict]:
    """Labeled test-split scans that no frozen set holds."""
    frozen = _frozen_ids(sets)
    return [r for r in records
            if r["split"] == "test" and r["label"] in ("confirmed", "corrected") and r["scanId"] not in frozen]


def _progress(candidates: list[dict]) -> tuple[int, int]:
    return len(candidates), len({r["printingId"] for r in candidates})


def freeze(records: list[dict], directory: Path = paths.TESTSETS, min_scans: int = MIN_SCANS,
           min_printings: int = MIN_PRINTINGS, today: str | None = None) -> dict:
    """Writes the next test-vN from the current pool, or raises FreezeError with the progress."""
    sets = frozen_sets(directory)
    candidates = sorted(pool(records, sets), key=lambda r: r["scanId"])
    scans, printings = _progress(candidates)
    if scans < min_scans or printings < min_printings:
        raise FreezeError(f"test pool has {scans}/{min_scans} labeled scans across {printings}/{min_printings} printings")
    version = sets[-1]["version"] + 1 if sets else 1
    frozen = {
        "name": f"test-v{version}",
        "version": version,
        "frozen": today or date.today().isoformat(),
        "scans": [{"scanId": r["scanId"], "printingId": r["printingId"]} for r in candidates],
        "printings": sorted({r["printingId"] for r in candidates}),
    }
    io.write_json(directory / f"test-v{version}.json", frozen)
    return frozen


def load(name: str, directory: Path = paths.TESTSETS) -> dict:
    """A frozen set by name ("test-v2") or the newest one ("latest")."""
    if name == "latest":
        sets = frozen_sets(directory)
        if not sets:
            raise FileNotFoundError(f"no frozen test set in {directory}; run `make freeze-test` once the pool is big enough")
        return sets[-1]
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no test set {name} in {directory}")
    return io.read_json(path)


def status_lines(all_records: list[dict], sets: list[dict], min_scans: int = MIN_SCANS,
                 min_printings: int = MIN_PRINTINGS) -> list[str]:
    """Label, split, and freeze progress for `prepare_dataset.py status`. `all_records` includes unlabeled scans."""
    labeled = [r for r in all_records if r["label"] in ("confirmed", "corrected")]
    confirmed = sum(r["label"] == "confirmed" for r in labeled)
    scans, printings = _progress(pool(all_records, sets))
    next_name = f"test-v{sets[-1]['version'] + 1 if sets else 1}"
    lines = [
        f"scans: {len(all_records)} imported, {len(labeled)} labeled ({confirmed} confirmed, "
        f"{len(labeled) - confirmed} corrected), {len(all_records) - len(labeled)} unlabeled",
        f"labeled split: {sum(r['split'] == 'train' for r in labeled)} train, {sum(r['split'] == 'test' for r in labeled)} test",
        f"test pool for {next_name}: {scans}/{min_scans} scans across {printings}/{min_printings} printings",
    ]
    for s in sets:
        lines.append(f"frozen: {s['name']} ({s['frozen']}, {len(s['scans'])} scans, {len(s['printings'])} printings)")
    return lines
```
(`status_lines` counts the split over *labeled* scans only, so in the test data record `c`, which is unlabeled, is excluded. That gives 1 train and 2 test: `a` and `d` are test, `b` is train.)

In `dataset.py`, `main`:
- Add `from . import testsets` **inside `main`** (not at module top), so `testsets` can import `dataset` without a cycle.
- Add the parsers: `sub.add_parser("status", help="labels, split, and test-set freeze progress")` and `sub.add_parser("freeze-test", help="freeze the next real-scan test set once the pool is big enough")`.
- Add the branches:
```python
    elif args.command == "status":
        for line in testsets.status_lines(scan_records(labeled_only=False), testsets.frozen_sets()):
            print(line)
    elif args.command == "freeze-test":
        try:
            frozen = testsets.freeze(scan_records())
        except testsets.FreezeError as error:
            raise SystemExit(f"not frozen: {error}")
        print(f"froze {frozen['name']}: {len(frozen['scans'])} scans across {len(frozen['printings'])} printings "
              f"-> {paths.TESTSETS.relative_to(paths.REPO)}/{frozen['name']}.json (commit it)")
```
Create an empty `ml/testsets/.gitkeep`.

`Makefile`: add both targets to `.PHONY`, and after `eval` add:
```make
## Scan labels, train/test split, and progress toward the next frozen test set.
status:
	cd ml && uv run scripts/prepare_dataset.py status

## Freeze the next real-scan test set (needs ≥200 labeled test scans across ≥30 printings).
freeze-test:
	cd ml && uv run scripts/prepare_dataset.py freeze-test
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

Run: `make status`
Expected (with no scans imported yet), four lines, ending with `test pool for test-v1: 0/200 scans across 0/30 printings`. Run `make freeze-test` and expect a `not frozen: test pool has 0/200 …` message with exit code 1. Both commands must run without a traceback.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/testsets.py ml/oplab/paths.py ml/oplab/dataset.py ml/testsets/.gitkeep Makefile ml/tests/test_testsets.py
git commit -m "Freeze versioned real-scan test sets and report label/split progress"
```

---

### Task 3: Wilson confidence intervals on headline metrics

**Files:**
- Modify: `ml/oplab/metrics.py` (`wilson`, `rate_ci`, `summarize`)
- Modify: `ml/oplab/evaluate.py` (`RESULT_COLUMNS`, `render_report`, the `append_result` call in `main`)
- Modify: `ml/tests/test_metrics.py`, `ml/tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `metrics.wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]`, rounded to 4 decimals and clamped to [0, 1], returning `(None, None)` when `n == 0`
  - `metrics.rate_ci(values: list) -> tuple[float | None, list]`, which returns `(rate, [low, high])` over the non-None values
  - `summarize(...)["summary"]` gains `detection_ci`, `top1_ci`, `ocr_accuracy_ci` and `within_group_ci`, each `[low, high]`
  - `evaluate.RESULT_COLUMNS` = the current columns plus `testset`, `top1_low`, `top1_high`, `ocr_accuracy_low`, `ocr_accuracy_high`, `within_group_low`, `within_group_high`, in that order and at the end. Task 4 fills `testset`.

- [ ] **Step 1: Write the failing tests**

Append to `ml/tests/test_metrics.py`:
```python
def test_wilson_interval():
    assert metrics.wilson(76, 100) == (0.6677, 0.8331)
    assert metrics.wilson(10, 10) == (0.7225, 1.0)
    assert metrics.wilson(0, 10) == (0.0, 0.2775)
    assert metrics.wilson(0, 0) == (None, None)


def test_rate_ci_ignores_none():
    assert metrics.rate_ci([True, False, None, True]) == (round(2 / 3, 4), list(metrics.wilson(2, 3)))
    assert metrics.rate_ci([None]) == (None, [None, None])


def test_summary_carries_intervals():
    predictions = [prediction("q1", "A", ocr="A", method="ocr-unique", group_size=1),
                   prediction("q2", "B", method="vision-only")]
    s = metrics.summarize(predictions, {**truth("q1", "A"), **truth("q2", "A_p1")}, lambda p: ATTRS[p])["summary"]
    assert s["top1"] == 0.5 and s["top1_ci"] == list(metrics.wilson(1, 2))
    assert s["detection_ci"] == list(metrics.wilson(2, 2))
    assert s["ocr_accuracy_ci"] == list(metrics.wilson(1, 2))
    assert s["within_group_ci"] == [None, None]
```
Append to `ml/tests/test_evaluate.py`:
```python
def test_result_columns_end_with_testset_and_intervals():
    assert evaluate.RESULT_COLUMNS[-7:] == ["testset", "top1_low", "top1_high", "ocr_accuracy_low",
                                           "ocr_accuracy_high", "within_group_low", "within_group_high"]


def test_report_shows_intervals():
    summary = {"n": 2, "detection": 1.0, "top1": 0.5, "top3": 0.5, "top1_given_detected": 0.5, "card_top1": 0.5,
               "variant_top1": None, "ocr_used": 0.5, "ocr_accuracy": 0.5, "within_group": None, "median_ms": 10.0,
               "detection_ci": [0.3424, 1.0], "top1_ci": [0.0946, 0.9054], "ocr_accuracy_ci": [0.0946, 0.9054],
               "within_group_ci": [None, None]}
    meta = {"backend": "vision-featureprint-r2", "rows": ["A", "B"]}
    report = evaluate.render_report("run", meta, {"summary": summary, "groups": {}, "hard_cases": []}, 0)
    assert "| Top-1 printing | 50.0% (9.5–90.5%) |" in report
    assert "| Within-group top-1 (right code, ≥2 printings) | – |" in report
```
The existing `test_append_result_migrates_old_header` builds its old header by dropping only `ocr_accuracy` and `within_group` from `RESULT_COLUMNS`. Change that line to drop every column from `ocr_accuracy` onward: `old_columns = evaluate.RESULT_COLUMNS[:evaluate.RESULT_COLUMNS.index("ocr_accuracy")]`. That keeps the test meaning "the pre-Phase-1 header".

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_metrics.py tests/test_evaluate.py`
Expected: FAIL (`wilson` / `rate_ci` missing, column list and report text mismatch).

- [ ] **Step 3: Implement**

In `metrics.py`, add `import math`, and above `summarize`:
```python
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion; (None, None) with no observations."""
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def rate_ci(values: list) -> tuple[float | None, list]:
    """(rate, [low, high]) over the non-None booleans."""
    values = [v for v in values if v is not None]
    if not values:
        return None, [None, None]
    return round(sum(values) / len(values), 4), list(wilson(sum(values), len(values)))
```
In `summarize`, build the four headline rates with `rate_ci` and add the intervals. `rate` stays for the other metrics.
```python
    detection, detection_ci = rate_ci([r["detected"] for r in rows])
    top1, top1_ci = rate_ci([r["top1"] for r in rows])
    ocr_accuracy, ocr_accuracy_ci = rate_ci([r["ocr_correct"] for r in detected])
    within_group, within_group_ci = rate_ci([r["top1"] for r in detected if r["method"] == "ocr+vision" and r["ocr_correct"]])
```
Use these variables for `"detection"`, `"top1"`, `"ocr_accuracy"` and `"within_group"` in the summary dict, and add `"detection_ci": detection_ci, "top1_ci": top1_ci, "ocr_accuracy_ci": ocr_accuracy_ci, "within_group_ci": within_group_ci`.

In `evaluate.py`:
- Extend `RESULT_COLUMNS` with `"testset", "top1_low", "top1_high", "ocr_accuracy_low", "ocr_accuracy_high", "within_group_low", "within_group_high"`.
- In `render_report`, add a helper next to `pct`:
```python
    def with_ci(value, ci):
        if value is None or ci is None or ci[0] is None:
            return pct(value)
        return f"{pct(value)} ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"
```
  and use `with_ci(s['detection'], s.get('detection_ci'))` for Detection. Do the same for Top-1 printing, OCR accuracy and Within-group. Add one line under the table: `"95% Wilson confidence intervals in parentheses.", ""`.
- In `main`'s `append_result({...})`, add:
```python
        "top1_low": result["summary"]["top1_ci"][0], "top1_high": result["summary"]["top1_ci"][1],
        "ocr_accuracy_low": result["summary"]["ocr_accuracy_ci"][0], "ocr_accuracy_high": result["summary"]["ocr_accuracy_ci"][1],
        "within_group_low": result["summary"]["within_group_ci"][0], "within_group_high": result["summary"]["within_group_ci"][1],
```
  `**result["summary"]` also passes the `*_ci` list keys. `append_result` already writes only `RESULT_COLUMNS` keys, so they're ignored there. Keep it that way.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/metrics.py ml/oplab/evaluate.py ml/tests/test_metrics.py ml/tests/test_evaluate.py
git commit -m "Report 95% Wilson intervals for headline recognition metrics"
```

---

### Task 4: Evaluate on a frozen test set, docs, and the v0 procedure

**Files:**
- Modify: `ml/oplab/testsets.py` (`entries`, `MissingScans`)
- Modify: `ml/oplab/evaluate.py` (`--testset`, loading entries, the `testset` result column)
- Modify: `ml/tests/test_testsets.py`
- Modify: `ml/README.md` (the "Real data" section and the Scripts table)
- Modify: `docs/cv-pipeline.md` (the "Scan logs" section: one paragraph)

**Interfaces:**
- Consumes:
  - `testsets.load` and the frozen set's JSON shape (Task 2)
  - `dataset.card_id_of(printing_id, card_ids)` and `dataset.catalog_card_ids()`, both existing
  - the `testset` result column (Task 3)
- Produces:
  - `class testsets.MissingScans(Exception)`
  - `testsets.entries(testset: dict, scans_dir: Path = paths.SCANS, card_ids: dict[str, str] | None = None) -> list[dict]`: manifest-style entries (`id`, `path`, `printingId`, `cardId`, `mode: "card"`, `tags: {"source": "scan", "testset": name}`)
  - `evaluate.py --testset NAME`, where `NAME` is `test-vN` or `latest`

- [ ] **Step 1: Write the failing tests** (append to `ml/tests/test_testsets.py`)

```python
FROZEN = {"name": "test-v1", "version": 1, "frozen": "2026-10-01",
          "scans": [{"scanId": "a", "printingId": "OP09-078-r1"}, {"scanId": "b", "printingId": "OP01-003"}],
          "printings": ["OP01-003", "OP09-078-r1"]}


def make_crops(root, *scan_ids):
    for scan_id in scan_ids:
        (root / scan_id).mkdir(parents=True)
        (root / scan_id / "crop.jpg").write_bytes(b"jpeg")


def test_testset_entries_use_frozen_truth(tmp_path):
    make_crops(tmp_path, "a", "b")
    rows = testsets.entries(FROZEN, tmp_path, card_ids={"OP09-078-r1": "OP09-078"})
    assert rows[0] == {"id": "scan:a", "path": str(tmp_path / "a" / "crop.jpg"), "printingId": "OP09-078-r1",
                       "cardId": "OP09-078", "mode": "card", "tags": {"source": "scan", "testset": "test-v1"}}
    assert rows[1]["printingId"] == "OP01-003" and rows[1]["cardId"] == "OP01-003"


def test_testset_entries_fail_on_missing_crops(tmp_path):
    make_crops(tmp_path, "a")
    with pytest.raises(testsets.MissingScans, match="1 of 2 scans in test-v1 have no crop.*b"):
        testsets.entries(FROZEN, tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_testsets.py`
Expected: FAIL (`entries` / `MissingScans` missing).

- [ ] **Step 3: Implement**

In `testsets.py`, add `from . import dataset` to the imports (there's no circular import: `dataset` imports `testsets` inside `main` only). If a cycle does appear at import time, import `dataset` inside `entries` instead. Then add:
```python
class MissingScans(Exception):
    pass


def entries(testset: dict, scans_dir: Path = paths.SCANS, card_ids: dict[str, str] | None = None) -> list[dict]:
    """Eval entries for a frozen set: each scan's crop with the printing frozen as its truth.
    Refuses when any crop is missing, so a partial import can't quietly shrink the test set."""
    missing = [s["scanId"] for s in testset["scans"] if not (scans_dir / s["scanId"] / "crop.jpg").exists()]
    if missing:
        raise MissingScans(f"{len(missing)} of {len(testset['scans'])} scans in {testset['name']} have no crop in "
                           f"{scans_dir}: {', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")
    return [{
        "id": f"scan:{s['scanId']}",
        "path": str(scans_dir / s["scanId"] / "crop.jpg"),
        "printingId": s["printingId"],
        "cardId": dataset.card_id_of(s["printingId"], card_ids),
        "mode": "card",
        "tags": {"source": "scan", "testset": testset["name"]},
    } for s in testset["scans"]]
```
In `evaluate.py`:
- Add `testsets` to the `from . import …` line.
- Add the argument `parser.add_argument("--testset", help="score a frozen real-scan test set (test-vN or 'latest') instead of the synthetic/photo manifest")`.
- Replace
```python
    if not paths.TEST_MANIFEST.exists():
        raise SystemExit("no test set; run prepare_dataset.py build-test first")
```
  and the following `entries = io.read_json(paths.TEST_MANIFEST)` with:
```python
    testset_name = ""
    if args.testset:
        try:
            testset = testsets.load(args.testset)
            entries = testsets.entries(testset, card_ids=dataset.catalog_card_ids())
        except (FileNotFoundError, testsets.MissingScans) as error:
            raise SystemExit(str(error))
        testset_name = testset["name"]
    else:
        if not paths.TEST_MANIFEST.exists():
            raise SystemExit("no test set; run prepare_dataset.py build-test first")
        entries = io.read_json(paths.TEST_MANIFEST)
```
  (The `meta`/`indexed` lines stay where they are.)
- In `render_report`'s first header lines, when a test set is used, the title should say so. Add a keyword parameter `testset: str = ""` to `render_report`, and, if it's set, append ``f" Test set `{testset}`."`` to the sentence that reports the test image count. Pass `testset=testset_name` from `main`.
- In `append_result({...})`, add `"testset": testset_name`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

Run: `cd ml && uv run scripts/evaluate.py --testset latest --name smoke`
Expected: it exits with `no frozen test set in …/ml/testsets; run \`make freeze-test\` once the pool is big enough`, **without** writing a run or a `results.csv` row. Check with `git status ml/results/results.csv`: it must be unchanged.

- [ ] **Step 5: Document**

In `ml/README.md`, replace the "**Device scan logs:**" bullet in "## Real data (the test set that matters)" with:
```markdown
- **Device scan logs:** the app logs every scan to Documents/Scans with the answer you gave on the
  phone: ✓ = `confirmed`, choosing another printing = `corrected`, no answer = `none`. Only labeled
  scans are data. Copy the folder to the Mac (Finder > iPhone > Files > OnePieceAR), then:
  ```sh
  uv run scripts/prepare_dataset.py import-scans ~/Downloads/Scans   # new scans + refreshed labels
  make status                                                          # labels, split, freeze progress
  ```
  **Split by printing, not by scan:** a stable hash sends ~30% of printings to *test* forever, so no
  photo of a test card is ever used for training (no leakage). Test scans wait in a pool; once it holds
  ≥200 labeled scans across ≥30 printings, `make freeze-test` writes `ml/testsets/test-vN.json` —
  commit it. Every model is then scored on the same frozen set:
  ```sh
  uv run scripts/evaluate.py --testset latest --name <model>
  ```
  Scans arriving later go to the pool for the next version, never into a frozen set. Train-split scans
  can become extra references: `generate_embeddings.py --with-scans labeled`.

  **v0 baseline:** right after freezing `test-v1`, record the Vision feature print on it, before any
  fine-tuned model: `uv run scripts/evaluate.py --testset test-v1 --name v0`.
```
In the metrics list of the walkthrough, add: `- Top-1, OCR accuracy, detection, and within-group show a 95% Wilson confidence interval: with 200 test scans at ~75% accuracy it is about ±6 points, so smaller differences between models are noise.`

In the Scripts table, change the `prepare_dataset.py` row to `` `synth`, `negatives`, `import-scans`, `build-test`, `status`, `freeze-test` `` and the `evaluate.py` row to `metrics via \`cardvision match\` on the manifest or a frozen test set (\`--testset\`), report, results history`.

In `docs/cv-pipeline.md`'s "Scan logs" section, after the sentence ending "Only labeled scans are meant for training and testing.", add: `The ML lab splits labeled scans by printing (≈30% of printings are test-only) and scores models on frozen real-scan test sets; see ml/README.md, "Real data".`

- [ ] **Step 6: Commit**

```bash
git add ml/oplab/testsets.py ml/oplab/evaluate.py ml/tests/test_testsets.py ml/README.md docs/cv-pipeline.md
git commit -m "Evaluate on frozen real-scan test sets and document the labeled-data workflow"
```
