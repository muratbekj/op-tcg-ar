"""Evaluate recognition on the fixed test set, using the device pipeline via `cardvision match`.

Writes ml/runs/<timestamp>-<name>/{predictions.jsonl, metrics.json, report.md} and appends one row
per run to ml/results/results.csv, which is tracked in git so improvements are compared on the same
test set over time.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from . import cardvision, dataset, io, metrics, paths

RESULT_COLUMNS = ["timestamp", "name", "backend", "index", "index_printings", "sources", "ocr", "n", "skipped",
                  "detection", "top1", "top3", "top1_given_detected", "card_top1", "variant_top1",
                  "ocr_used", "median_ms", "negatives", "suggested_threshold"]


def printing_attributes() -> dict[str, dict]:
    attributes = {}
    if paths.FULL_CATALOG.exists():
        for p in io.read_json(paths.FULL_CATALOG):
            attributes[p["printingId"]] = {"set": p["set"], "rarity": p["rarity"], "kind": p["kind"]}
    for p in io.read_json(paths.DATA_CARDS / "printings.json"):
        attributes.setdefault(p["id"], {"set": p["cardId"].split("-")[0], "rarity": p["rarity"], "kind": p["kind"]})
    return attributes


def variant_lookup() -> dict[str, str]:
    cards = {c["id"]: c for c in io.read_json(paths.DATA_CARDS / "cards.json")}
    return {
        p["id"]: p.get("variantId") or cards[p["cardId"]]["defaultVariantId"]
        for p in io.read_json(paths.DATA_CARDS / "printings.json") if p["cardId"] in cards
    }


def render_report(name: str, meta: dict, result: dict, skipped: int) -> str:
    s = result["summary"]
    pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"
    lines = [
        f"# Recognition eval: {name}", "",
        f"Backend `{meta['backend']}`, {len(set(meta['rows']))} printings in index ({len(meta['rows'])} rows). "
        f"{s['n']} test images; {skipped} skipped because their printing isn't in the index.", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Detection | {pct(s['detection'])} |",
        f"| Top-1 printing | {pct(s['top1'])} |",
        f"| Top-3 printing | {pct(s['top3'])} |",
        f"| Top-1 given detected | {pct(s['top1_given_detected'])} |",
        f"| Top-1 card number | {pct(s['card_top1'])} |",
        f"| Top-1 variant (what spawns) | {pct(s['variant_top1'])} |",
        f"| OCR used | {pct(s['ocr_used'])} |",
        f"| OCR accuracy when used | {pct(s['ocr_accuracy'])} |",
        f"| Median latency (Mac) | {s['median_ms']} ms |", "",
    ]
    for key, values in result["groups"].items():
        has_precision = any("precision" in v for v in values.values())
        lines += [f"## By {key}", "", "| Value | n | Recall |" + (" Precision |" if has_precision else ""),
                  "| --- | --- | --- |" + (" --- |" if has_precision else "")]
        for value, v in values.items():
            lines.append(f"| {value} | {v['n']} | {pct(v['recall'])} |" + (f" {pct(v.get('precision'))} |" if has_precision else ""))
        lines.append("")
    rejection = result.get("rejection")
    if rejection and rejection["negatives"]:
        lines += [f"## Rejecting unknown cards ({rejection['negatives']} non-roster photos)", "",
                  f"Suggested minimum similarity (≤5% false accepts): **{rejection['suggested_threshold']}**", "",
                  "| Threshold | Correct kept | False accepts |", "| --- | --- | --- |"]
        for point in rejection["curve"]:
            lines.append(f"| {point['threshold']} | {pct(point['correct_kept'])} | {pct(point['false_accept'])} |")
        lines.append("")
    lines += ["## Hardest misses", "", "| Image | Expected | Predicted | Similarity |", "| --- | --- | --- | --- |"]
    for case in result["hard_cases"][:15]:
        similarity = "–" if case["similarity"] is None else f"{case['similarity']:.3f}"
        lines.append(f"| {case['id']} | {case['expected']} | {case['predicted'] or 'not detected'} | {similarity} |")
    return "\n".join(lines) + "\n"


def append_result(row: dict) -> None:
    paths.RESULTS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not paths.RESULTS.exists()
    with paths.RESULTS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="baseline", help="label for this run in results.csv")
    parser.add_argument("--index", type=Path, default=paths.INDEX)
    parser.add_argument("--model", type=Path, help="Core ML embedder; must match the index backend")
    parser.add_argument("--source", action="append", choices=["photo", "synth", "scan", "negative"],
                        help="restrict to these sources (repeatable); default all")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--limit", type=int, help="evaluate only the first N images (quick checks)")
    args = parser.parse_args(argv)

    if not paths.TEST_MANIFEST.exists():
        raise SystemExit("no test set; run prepare_dataset.py build-test first")
    meta = io.read_json(cardvision.meta_path(args.index))
    indexed = set(meta["rows"])
    entries = io.read_json(paths.TEST_MANIFEST)
    if args.source:
        entries = [e for e in entries if e["tags"]["source"] in args.source]
    negatives = [e for e in entries if e["printingId"] == dataset.NEGATIVE]
    in_index = [e for e in entries if e["printingId"] in indexed]
    skipped = len(entries) - len(in_index) - len(negatives)
    if args.limit:
        in_index = in_index[: args.limit]
        negatives = negatives[: args.limit]
    if not in_index:
        raise SystemExit("no test images whose printing is in the index")

    print(f"evaluating {len(in_index)} images against {len(indexed)} printings ({meta['backend']}) …")
    predictions = []
    for mode in ("photo", "card"):
        queries = [{"id": e["id"], "path": e["path"]} for e in in_index if e["mode"] == mode]
        predictions += cardvision.match(args.index, queries, mode, ocr=not args.no_ocr, model=args.model)

    attributes = printing_attributes()
    variants = variant_lookup()
    truth = {e["id"]: e for e in in_index}
    result = metrics.summarize(predictions, truth, lambda pid: attributes.get(pid, {}), variants.get)
    negative_predictions = cardvision.match(
        args.index, [{"id": e["id"], "path": e["path"]} for e in negatives], "photo", ocr=not args.no_ocr, model=args.model)
    curve = metrics.rejection_curve(result["rows"], negative_predictions)
    result["rejection"] = {"negatives": len(negative_predictions), "curve": curve,
                           "suggested_threshold": metrics.suggest_threshold(curve)}
    predictions += negative_predictions

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = paths.RUNS / f"{stamp}-{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
    io.write_json(run_dir / "metrics.json", {k: result[k] for k in ("summary", "groups", "hard_cases", "rejection")})
    report = render_report(args.name, meta, result, skipped)
    (run_dir / "report.md").write_text(report)

    append_result({
        "timestamp": stamp, "name": args.name, "backend": meta["backend"],
        "index": args.index.name, "index_printings": len(indexed),
        "sources": "+".join(sorted({e["tags"]["source"] for e in in_index})),
        "ocr": not args.no_ocr, "skipped": skipped, **result["summary"],
        "negatives": len(negative_predictions), "suggested_threshold": result["rejection"]["suggested_threshold"],
    })
    print(report)
    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
