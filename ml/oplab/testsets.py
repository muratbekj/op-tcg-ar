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
