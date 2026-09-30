"""Generate reference embeddings with the device's own pipeline (via the `cardvision` CLI).

Full-catalog index (default) -> data/cards/printings.f32 + printings.meta.json, bundled into the app.
Every card in the catalog, plus the roster's own clean art (app's Resources/Cards/<id>.*).
Each printing can have several reference rows:
- API art (data/cards/art/<id>.jpg), which carries a SAMPLE watermark
- your own clean scan in the app's Resources/Cards/<id>.* (if it differs from the API art)
- with --with-scans: confirmed device scans from the reference split (the learning loop)
Identical images are embedded once (deduplicated by content digest).
Needs ~1.4 GB: run fetch_cards.py --art all first.

Roster-only index (--scope roster) -> ml/datasets/references/roster.f32: roster printings only,
for development and testing.
"""

import argparse
import hashlib
from pathlib import Path

from . import cardvision, dataset, io, paths


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_references(with_scans: str) -> list[dict]:
    entries, seen = [], set()

    def add(printing_id: str, path: Path, source: str) -> None:
        digest = _digest(path)
        if digest not in seen:
            seen.add(digest)
            entries.append({"printingId": printing_id, "path": str(path), "source": source})

    for printing in io.read_json(paths.DATA_CARDS / "printings.json"):
        art = paths.ART / f"{printing['id']}.jpg"
        if art.exists():
            add(printing["id"], art, "api")
        for own in sorted(paths.APP_CARDS.glob(f"{printing['id']}.*")):
            add(printing["id"], own, "app")

    if with_scans != "none":
        roster_ids = {e["printingId"] for e in entries}
        for record in dataset.scan_records():
            trusted = with_scans == "all" or record["label"] == "corrected"
            if record["split"] == "reference" and trusted and record["printingId"] in roster_ids:
                add(record["printingId"], Path(record["path"]), "scan")
    return entries


def full_references(with_scans: str = "none") -> list[dict]:
    """Roster references first (they may include your own clean art and scans), then the API art of
    every other catalog printing. Identical images are embedded once."""
    entries = roster_references(with_scans)
    seen = {_digest(Path(e["path"])) for e in entries}
    for printing in io.read_json(paths.FULL_CATALOG):
        art = paths.ART / f"{printing['printingId']}.jpg"
        if art.exists() and (digest := _digest(art)) not in seen:
            seen.add(digest)
            entries.append({"printingId": printing["printingId"], "path": str(art), "source": "api"})
    return entries


def update_embedding_rows(meta: dict) -> None:
    """Records each printing's first row in printings.json (informational; the app reads the meta file)."""
    first_row: dict[str, int] = {}
    for row, printing_id in enumerate(meta["rows"]):
        first_row.setdefault(printing_id, row)
    printings_path = paths.DATA_CARDS / "printings.json"
    printings = io.read_json(printings_path)
    for printing in printings:
        printing["embeddingRow"] = first_row.get(printing["id"])
    io.write_json(printings_path, printings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=["roster", "full"], default="full",
                        help="full (default): every catalog printing, bundled into the app; roster: roster printings only")
    parser.add_argument("--with-scans", choices=["none", "corrected", "all"], default="none",
                        help="add device scans from the reference split as extra rows")
    parser.add_argument("--model", type=Path, help="Core ML embedder (.mlpackage); default is the Vision feature print")
    parser.add_argument("--out", type=Path, help="output .f32 (default depends on scope)")
    parser.add_argument("--min-similarity", type=float,
                        help="device rejection threshold stored in the index metadata; take evaluate.py's "
                             "suggestion for this backend (default: the app's feature print default)")
    args = parser.parse_args(argv)

    if args.scope == "roster":
        entries = roster_references(args.with_scans)
        out = args.out or paths.REFERENCES / "roster.f32"
    else:
        entries = full_references(args.with_scans)
        out = args.out or paths.INDEX
    if not entries:
        raise SystemExit("no reference images found; run fetch_cards.py --art all first")

    sources: dict[str, int] = {}
    for entry in entries:
        sources[entry["source"]] = sources.get(entry["source"], 0) + 1
    print(f"embedding {len(entries)} references {sources} …")
    info = cardvision.embed([{"printingId": e["printingId"], "path": e["path"]} for e in entries], out, args.model,
                            args.min_similarity)
    print(f"  {info['rows']} rows × {info['dimension']} ({info['backend']}) -> {out}")

    if out == paths.INDEX:
        update_embedding_rows(io.read_json(cardvision.meta_path(out)))


if __name__ == "__main__":
    main()
