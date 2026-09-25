"""Recognition metrics. Pure functions over predictions (from `cardvision match`) and ground truth."""

from collections import defaultdict
from statistics import median
from typing import Callable

PRINTING_ATTRIBUTES = ("set", "rarity", "kind")


def top_ids(prediction: dict, k: int) -> list[str]:
    return [c["printingId"] for c in prediction.get("candidates", [])[:k]]


def summarize(
    predictions: list[dict],
    truth: dict[str, dict],
    attributes: Callable[[str], dict],
    variant_of: Callable[[str], str | None] | None = None,
) -> dict:
    """
    truth[id] = {"printingId", "cardId", "tags": {...}}; `attributes(printingId)` gives set/rarity/kind.
    `variant_of(printingId)` maps a printing to the character variant that would spawn, when known.
    """
    rows = []
    for prediction in predictions:
        expected = truth.get(prediction["id"])
        if expected is None:
            continue
        top = prediction["candidates"][0] if prediction.get("candidates") else None
        predicted = top["printingId"] if top else None
        expected_variant = variant_of(expected["printingId"]) if variant_of else None
        rows.append({
            "id": prediction["id"],
            "expected": expected["printingId"],
            "predicted": predicted,
            "detected": bool(prediction.get("detected")) and top is not None,
            "top1": predicted == expected["printingId"],
            "top3": expected["printingId"] in top_ids(prediction, 3),
            "card": top is not None and top["cardId"] == expected["cardId"],
            "variant": (variant_of(predicted) == expected_variant) if expected_variant and predicted else None,
            "similarity": top["similarity"] if top else None,
            "ocr": prediction.get("ocrCardId"),
            "ocr_correct": prediction.get("ocrCardId") == expected["cardId"] if prediction.get("ocrCardId") else None,
            "ms": prediction.get("ms"),
            "tags": {**expected.get("tags", {}), **attributes(expected["printingId"])},
            "candidates": prediction.get("candidates", [])[:5],
        })

    def rate(values: list) -> float | None:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    detected = [r for r in rows if r["detected"]]
    summary = {
        "n": len(rows),
        "detection": rate([r["detected"] for r in rows]),
        "top1": rate([r["top1"] for r in rows]),
        "top3": rate([r["top3"] for r in rows]),
        "top1_given_detected": rate([r["top1"] for r in detected]),
        "card_top1": rate([r["card"] for r in rows]),
        "variant_top1": rate([r["variant"] for r in rows]),
        "ocr_used": rate([r["ocr"] is not None for r in detected]),
        "ocr_accuracy": rate([r["ocr_correct"] for r in rows]),
        "median_ms": round(median(r["ms"] for r in rows if r["ms"] is not None), 1) if rows else None,
    }
    return {"summary": summary, "groups": group_metrics(rows, attributes), "hard_cases": hard_cases(rows), "rows": rows}


def group_metrics(rows: list[dict], attributes: Callable[[str], dict]) -> dict:
    """Per tag value: accuracy (recall) for every tag; precision too for printing attributes,
    counted over the predictions that landed in that group."""
    groups: dict[str, dict] = {}
    keys = sorted({key for row in rows for key in row["tags"]})
    for key in keys:
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, n] by true value
        predicted: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, n] by predicted value
        for row in rows:
            value = str(row["tags"].get(key))
            totals[value][0] += row["top1"]
            totals[value][1] += 1
            if key in PRINTING_ATTRIBUTES and row["predicted"]:
                predicted_value = str(attributes(row["predicted"]).get(key))
                predicted[predicted_value][0] += row["top1"]
                predicted[predicted_value][1] += 1
        groups[key] = {
            value: {
                "n": n,
                "recall": round(correct / n, 4),
                **({"precision": round(predicted[value][0] / predicted[value][1], 4) if predicted[value][1] else None}
                   if key in PRINTING_ATTRIBUTES else {}),
            }
            for value, (correct, n) in sorted(totals.items())
        }
    return groups


def hard_cases(rows: list[dict], limit: int = 50) -> list[dict]:
    """Misses, most confident first. Confident mistakes are the hard negatives worth mining."""
    misses = [r for r in rows if not r["top1"]]
    misses.sort(key=lambda r: -(r["similarity"] or -1))
    return [
        {k: r[k] for k in ("id", "expected", "predicted", "similarity", "detected", "ocr", "tags", "candidates")}
        for r in misses[:limit]
    ]


def rejection_curve(positive_rows: list[dict], negative_predictions: list[dict],
                    thresholds: tuple[float, ...] = tuple(round(0.5 + 0.025 * i, 3) for i in range(19))) -> list[dict]:
    """For each minimum similarity: the share of test images still recognized correctly, and the
    share of non-roster cards wrongly accepted as some roster printing."""
    negatives = [p["candidates"][0]["similarity"] for p in negative_predictions if p.get("candidates")]
    curve = []
    for threshold in thresholds:
        kept = [r for r in positive_rows if r["top1"] and r["similarity"] is not None and r["similarity"] >= threshold]
        accepted = [s for s in negatives if s >= threshold]
        curve.append({
            "threshold": threshold,
            "correct_kept": round(len(kept) / len(positive_rows), 4) if positive_rows else None,
            "false_accept": round(len(accepted) / len(negative_predictions), 4) if negative_predictions else None,
        })
    return curve


def suggest_threshold(curve: list[dict], max_false_accept: float = 0.05) -> float | None:
    """Lowest threshold whose false-accept rate is within budget."""
    for point in curve:
        if point["false_accept"] is not None and point["false_accept"] <= max_false_accept:
            return point["threshold"]
    return None
