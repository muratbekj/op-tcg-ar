"""Datasets: import device scan logs, generate synthetic photos, and assemble the fixed test set.

Sources, all under ml/datasets/raw/ (gitignored):
- scans/<scanId>/{crop.jpg, scan.json}  copied from the device's Documents/Scans
- photos/<printingId>/[<condition>/]*.jpg  your own photos of physical cards (the test set that matters)
- synth/<printingId>/*.jpg  synthetic photos from API art
- negatives/<printingId>/*.jpg  synthetic photos of cards outside the roster (should be rejected)

Scan logs are real data only once the user answered them on the phone: `label` is `confirmed` or
`corrected`. Unlabeled scans (`none`) are never used. Labeled scans are split by a stable hash of their
printing ID: ~30% of printings are test-only forever, so photos of the same physical card never land
on both sides. Test scans are scored through frozen test sets (`testsets.py`), never the manifest.
"""

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import io, optcg, paths, synth

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
TEST_FRACTION = 0.3
NEGATIVE = "none"  # printingId of test images that should not match anything


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
        scan_id = str(scan_id)
        if scan_id != record_path.parent.name:  # entries() resolves crops by folder name
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
            "method": record.get("method"),
            "split": printing_split(printing_id),
        })
    return records


def train_records(scans_dir: Path = paths.SCANS, testsets_dir: Path = paths.TESTSETS) -> list[dict]:
    """Labeled train-split scans that no frozen test set holds, so a relabeled frozen scan never trains."""
    from . import testsets  # here, not at module top: testsets imports dataset

    frozen = testsets._frozen_ids(testsets.frozen_sets(testsets_dir))
    return [r for r in scan_records(scans_dir) if r["split"] == "train" and r["scanId"] not in frozen]


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


def import_scans(source: Path, scans_dir: Path = paths.SCANS) -> dict:
    """Copies new scan folders from an exported Scans folder and refreshes scan.json of already
    imported ones (a scan can be confirmed or corrected on the phone after an earlier export).
    Crops never change on the device (only scan.json is rewritten when a scan is relabeled), so only
    scan.json is compared. Folders without crop.jpg or with an unreadable or incomplete scan.json are
    skipped, reported, and left where they are; an existing imported copy is never overwritten by one."""
    new, updated, skipped = 0, 0, []
    for record in sorted(source.glob("*/scan.json")):
        result = _import_folder(record.parent, scans_dir)
        if result == "skipped":
            skipped.append(record.parent.name)
        elif result == "new":
            new += 1
        elif result == "updated":
            updated += 1
    return {"new": new, "updated": updated, "skipped": skipped}


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


def generate_synth(printing_ids: list[str], per_printing: int, seed: int) -> int:
    """Renders synthetic photos for printings whose art is downloaded. Deterministic per seed."""
    count = 0
    for printing_id in printing_ids:
        art = paths.ART / f"{printing_id}.jpg"
        if not art.exists():
            continue
        card = Image.open(art)
        out_dir = paths.SYNTH / printing_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for index in range(per_printing):
            rng = np.random.default_rng([seed, index, int(hashlib.md5(printing_id.encode()).hexdigest()[:8], 16)])
            condition = synth.CONDITIONS[index % len(synth.CONDITIONS)]
            photo, tags = synth.render_photo(card, rng, condition)
            photo.save(out_dir / f"{index:03d}_{condition}.jpg", quality=90)
            io.write_json(out_dir / f"{index:03d}_{condition}.json", tags)
            count += 1
    return count


def generate_negatives(count: int, seed: int) -> int:
    """Synthetic photos of random non-roster printings. Recognition should reject these, and
    evaluate.py uses them to choose the minimum similarity for accepting a match."""
    roster_cards = {p["cardId"] for p in io.read_json(paths.DATA_CARDS / "printings.json")}
    rows = [r for r in optcg.fetch_rows(paths.API_CACHE) if r["card_set_id"] not in roster_cards]
    rng = np.random.default_rng(seed)
    chosen = [rows[i] for i in rng.choice(len(rows), size=min(count, len(rows)), replace=False)]
    downloaded, _, failed = optcg.download_art(chosen, paths.ART)
    if failed:
        print(f"  {len(failed)} art downloads failed")
    rendered = 0
    for index, row in enumerate(chosen):
        art = paths.ART / f"{row['card_image_id']}.jpg"
        if not art.exists():
            continue
        out_dir = paths.NEGATIVES / row["card_image_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        condition = synth.CONDITIONS[index % len(synth.CONDITIONS)]
        photo, _ = synth.render_photo(Image.open(art), np.random.default_rng([seed, index]), condition)
        photo.save(out_dir / f"000_{condition}.jpg", quality=90)
        rendered += 1
    return rendered


def negative_entries(negatives_dir: Path = paths.NEGATIVES) -> list[dict]:
    return [
        {"id": f"negative:{image.parent.name}/{image.name}", "path": str(image), "printingId": NEGATIVE,
         "mode": "photo", "tags": {"source": "negative", "condition": image.stem.split("_", 1)[1],
                                   "actual": image.parent.name}}
        for image in sorted(negatives_dir.glob("*/*.jpg"))
    ]


def photo_entries(photos_dir: Path = paths.PHOTOS) -> list[dict]:
    entries = []
    for image in sorted(photos_dir.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(photos_dir).parts
        condition = relative[1] if len(relative) > 2 else "unlabeled"
        entries.append({"id": f"photo:{'/'.join(relative)}", "path": str(image), "printingId": relative[0],
                        "mode": "photo", "tags": {"source": "photo", "condition": condition}})
    return entries


def synth_entries(synth_dir: Path = paths.SYNTH) -> list[dict]:
    entries = []
    for image in sorted(synth_dir.glob("*/*.jpg")):
        condition = image.stem.split("_", 1)[1]
        entries.append({"id": f"synth:{image.parent.name}/{image.name}", "path": str(image),
                        "printingId": image.parent.name, "mode": "photo",
                        "tags": {"source": "synth", "condition": condition}})
    return entries


def catalog_card_ids() -> dict[str, str]:
    """printingId -> cardId from the full catalog (API printing IDs are irregular, so don't split them)."""
    if not paths.FULL_CATALOG.exists():
        return {}
    return {p["printingId"]: p["cardId"] for p in io.read_json(paths.FULL_CATALOG) if p.get("cardId")}


def card_id_of(printing_id: str, card_ids: dict[str, str] | None = None) -> str:
    return (card_ids or {}).get(printing_id) or printing_id.split("_", 1)[0]


def build_test_manifest() -> list[dict]:
    """Diagnostics: your photos, synthetic photos, and negatives. Real scans are scored through frozen test sets (testsets.py)."""
    entries = photo_entries() + synth_entries() + negative_entries()
    card_ids = catalog_card_ids()
    for entry in entries:
        entry["cardId"] = NEGATIVE if entry["printingId"] == NEGATIVE else card_id_of(entry["printingId"], card_ids)
    return entries


def index_printings() -> set[str] | None:
    """Printing IDs in the recognition index, or None when its metadata file doesn't exist."""
    if not paths.INDEX_META.exists():
        return None
    return set(io.read_json(paths.INDEX_META)["rows"])


def main(argv: list[str] | None = None) -> None:
    from . import testsets  # here, not at module top: testsets imports dataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    scans = sub.add_parser("import-scans", help="copy an exported device Scans folder into datasets/raw/scans")
    scans.add_argument("source", type=Path)
    gen = sub.add_parser("synth", help="render synthetic photos from downloaded art")
    gen.add_argument("--scope", choices=["roster", "full"], default="roster")
    gen.add_argument("--per-printing", type=int, default=14)
    gen.add_argument("--seed", type=int, default=0)
    neg = sub.add_parser("negatives", help="render photos of random non-roster cards (downloads their art)")
    neg.add_argument("--count", type=int, default=150)
    neg.add_argument("--seed", type=int, default=0)
    inbox = sub.add_parser("import-inbox", help="Mac mini: import everything copied into ~/oplab-inbox and archive it")
    inbox.add_argument("--inbox", type=Path, default=paths.INBOX)
    sub.add_parser("build-test", help="assemble datasets/test/manifest.json")
    sub.add_parser("status", help="labels, split, and test-set freeze progress")
    sub.add_parser("freeze-test", help="freeze the next real-scan test set once the pool is big enough")
    args = parser.parse_args(argv)

    if args.command == "import-scans":
        result = import_scans(args.source)
        print(f"imported {result['new']} new scans, refreshed {result['updated']} labels")
        if result["skipped"]:
            print(f"  skipped {len(result['skipped'])} folders without crop.jpg or with an unreadable scan.json: "
                  f"{', '.join(result['skipped'][:10])}")
        if not any(args.source.glob("*/scan.json")):
            print(f"  no <scanId>/scan.json found under {args.source}; the source should be the Scans folder itself")
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
    elif args.command == "synth":
        if args.scope == "roster":
            ids = [p["id"] for p in io.read_json(paths.DATA_CARDS / "printings.json")]
        else:
            ids = [p["printingId"] for p in io.read_json(paths.FULL_CATALOG)]
        print(f"rendered {generate_synth(ids, args.per_printing, args.seed)} synthetic photos")
    elif args.command == "negatives":
        print(f"rendered {generate_negatives(args.count, args.seed)} negative photos")
    elif args.command == "build-test":
        entries = build_test_manifest()
        io.write_json(paths.TEST_MANIFEST, entries)
        by_source: dict[str, int] = {}
        for entry in entries:
            by_source[entry["tags"]["source"]] = by_source.get(entry["tags"]["source"], 0) + 1
        print(f"test set: {len(entries)} images {by_source} -> {paths.TEST_MANIFEST}")
    elif args.command == "status":
        indexed = index_printings()
        for line in testsets.status_lines(scan_records(labeled_only=False), testsets.frozen_sets(), indexed=indexed):
            print(line)
        if indexed is None:
            print("index missing: can't tell which scans are freezable (build it: generate_embeddings.py)")
        from . import shipped

        print(shipped.status_line(len(scan_records())))
    elif args.command == "freeze-test":
        indexed = index_printings()
        if indexed is None:
            raise SystemExit("not frozen: build the full-catalog index first: generate_embeddings.py")
        try:
            frozen = testsets.freeze(scan_records(), indexed=indexed)
        except testsets.FreezeError as error:
            raise SystemExit(f"not frozen: {error}")
        print(f"froze {frozen['name']}: {len(frozen['scans'])} scans across {len(frozen['printings'])} printings "
              f"-> {paths.TESTSETS.relative_to(paths.REPO)}/{frozen['name']}.json (commit it)")


if __name__ == "__main__":
    main()
