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
                  "ocr_used", "median_ms", "negatives", "suggested_threshold", "ocr_accuracy", "within_group",
                  "testset", "top1_low", "top1_high", "ocr_accuracy_low", "ocr_accuracy_high",
                  "within_group_low", "within_group_high"]


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
    def with_ci(value, ci):
        if value is None or ci is None or ci[0] is None:
            return pct(value)
        return f"{pct(value)} ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"

    lines = [
        f"# Recognition eval: {name}", "",
        f"Backend `{meta['backend']}`, {len(set(meta['rows']))} printings in index ({len(meta['rows'])} rows). "
        f"{s['n']} test images; {skipped} skipped because their printing isn't in the index.", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Detection | {with_ci(s['detection'], s.get('detection_ci'))} |",
        f"| Top-1 printing | {with_ci(s['top1'], s.get('top1_ci'))} |",
        f"| Top-3 printing | {pct(s['top3'])} |",
        f"| Top-1 given detected | {pct(s['top1_given_detected'])} |",
        f"| Top-1 card number | {pct(s['card_top1'])} |",
        f"| Top-1 variant (what spawns) | {pct(s['variant_top1'])} |",
        f"| OCR read a catalog code | {pct(s['ocr_used'])} |",
        f"| OCR accuracy | {with_ci(s['ocr_accuracy'], s.get('ocr_accuracy_ci'))} |",
        f"| Within-group top-1 (right code, ≥2 printings) | {with_ci(s['within_group'], s.get('within_group_ci'))} |",
        f"| Median latency (Mac) | {s['median_ms']} ms |", "",
        "95% Wilson confidence intervals in parentheses.", "",
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
    """Appends one run. An older header is migrated by rewriting the file; old rows keep their values."""
    paths.RESULTS.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] | None = []
    if paths.RESULTS.exists():
        with paths.RESULTS.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            if reader.fieldnames == RESULT_COLUMNS:
                existing = None  # header is current: just append
    mode = "a" if existing is None else "w"
    with paths.RESULTS.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if existing is not None:
            writer.writeheader()
            for old in existing:
                writer.writerow({k: old.get(k, "") for k in RESULT_COLUMNS})
        writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})


def interval_columns(summary: dict) -> dict:
    """Flatten the summary's [low, high] intervals into results.csv columns."""
    return {f"{metric}_{bound}": summary[f"{metric}_ci"][i]
            for metric in ("top1", "ocr_accuracy", "within_group") for i, bound in enumerate(("low", "high"))}


def split_negatives(negatives: list[dict], indexed: set[str],
                    card_ids: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Negatives are photos of cards outside the roster. When the index covers their printing (a
    full-catalog index), they are cards the app should identify, so they become labeled positives.
    Only negatives outside the index remain for the rejection curve."""
    positives, true_negatives = [], []
    for entry in negatives:
        actual = entry["tags"].get("actual")
        if actual in indexed:
            positives.append({**entry, "printingId": actual, "cardId": dataset.card_id_of(actual, card_ids),
                              "tags": {**entry["tags"], "source": "catalog"}})
        else:
            true_negatives.append(entry)
    return positives, true_negatives


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
    catalog_positives, negatives = split_negatives(
        [e for e in entries if e["printingId"] == dataset.NEGATIVE], indexed, dataset.catalog_card_ids())
    in_index = [e for e in entries if e["printingId"] in indexed] + catalog_positives
    skipped = len(entries) - len(in_index) - len(negatives)
    if args.limit:
        in_index = in_index[: args.limit]
        negatives = negatives[: args.limit]
    if not in_index:
        raise SystemExit("no test images whose printing is in the index")

    print(f"evaluating {len(in_index)} images against {len(indexed)} printings ({meta['backend']}) …")
    catalog = paths.FULL_CATALOG if paths.FULL_CATALOG.exists() else None
    predictions = []
    for mode in ("photo", "card"):
        queries = [{"id": e["id"], "path": e["path"]} for e in in_index if e["mode"] == mode]
        predictions += cardvision.match(args.index, queries, mode, ocr=not args.no_ocr, model=args.model,
                                    catalog=catalog)

    attributes = printing_attributes()
    variants = variant_lookup()
    truth = {e["id"]: e for e in in_index}
    result = metrics.summarize(predictions, truth, lambda pid: attributes.get(pid, {}), variants.get)
    negative_predictions = cardvision.match(
        args.index, [{"id": e["id"], "path": e["path"]} for e in negatives], "photo", ocr=not args.no_ocr, model=args.model,
        catalog=catalog)
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
        **interval_columns(result["summary"]),
        "negatives": len(negative_predictions), "suggested_threshold": result["rejection"]["suggested_threshold"],
    })
    print(report)
    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
