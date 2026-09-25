"""Datasets: import device scan logs, generate synthetic photos, and assemble the fixed test set.

Sources, all under ml/datasets/raw/ (gitignored):
- scans/<scanId>/{crop.jpg, scan.json}  copied from the device's Documents/Scans
- photos/<printingId>/[<condition>/]*.jpg  your own photos of physical cards (the test set that matters)
- synth/<printingId>/*.jpg  synthetic photos from API art
- negatives/<printingId>/*.jpg  synthetic photos of cards outside the roster (should be rejected)

Scan logs are split by a stable hash of the scan ID: 30% join the test set, the rest are available
as extra references (the learning loop). The split never changes as new scans arrive.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import io, optcg, paths, synth

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
TEST_FRACTION = 0.3
NEGATIVE = "none"  # printingId of test images that should not match anything


def scan_split(scan_id: str) -> str:
    bucket = int(hashlib.sha256(scan_id.encode()).hexdigest(), 16) % 100
    return "test" if bucket < TEST_FRACTION * 100 else "reference"


def scan_records(scans_dir: Path = paths.SCANS) -> list[dict]:
    """Each logged scan with its label. Uncorrected scans are weak labels: the user saw the top
    guess and didn't object, which is usually but not always right."""
    records = []
    for record_path in sorted(scans_dir.glob("*/scan.json")):
        record = io.read_json(record_path)
        crop = record_path.parent / "crop.jpg"
        if not crop.exists():
            continue
        records.append({
            "id": f"scan:{record['id']}",
            "path": str(crop),
            "printingId": record["finalPrintingID"],
            "label": "corrected" if record.get("corrected") else "weak",
            "split": scan_split(record["id"]),
        })
    return records


def import_scans(source: Path) -> int:
    """Copies an exported Scans folder into datasets/raw/scans, skipping ones already imported."""
    count = 0
    for record in sorted(source.glob("*/scan.json")):
        target = paths.SCANS / record.parent.name
        if not target.exists():
            shutil.copytree(record.parent, target)
            count += 1
    return count


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
    optcg.download_art(chosen, paths.ART)
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


def build_test_manifest() -> list[dict]:
    """Photos (all), synthetic photos (all), and the test split of scans (already rectified crops)."""
    entries = photo_entries() + synth_entries() + negative_entries()
    for record in scan_records():
        if record["split"] == "test":
            entries.append({"id": record["id"], "path": record["path"], "printingId": record["printingId"],
                            "mode": "card", "tags": {"source": "scan", "label": record["label"]}})
    for entry in entries:
        entry["cardId"] = NEGATIVE if entry["printingId"] == NEGATIVE else entry["printingId"].split("_", 1)[0]
    return entries


def main(argv: list[str] | None = None) -> None:
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
    sub.add_parser("build-test", help="assemble datasets/test/manifest.json")
    args = parser.parse_args(argv)

    if args.command == "import-scans":
        print(f"imported {import_scans(args.source)} scans")
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


if __name__ == "__main__":
    main()
