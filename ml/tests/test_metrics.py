from oplab import metrics


def prediction(id, *printing_ids, detected=True, ocr=None):
    return {"id": id, "detected": detected, "ocrCardId": ocr, "ms": 10.0,
            "candidates": [{"printingId": p, "cardId": p.split("_")[0], "similarity": 0.9 - i * 0.1, "matchesOCR": False}
                           for i, p in enumerate(printing_ids)]}


ATTRS = {"A": {"set": "OP05", "rarity": "SEC", "kind": "base"},
         "A_p1": {"set": "OP05", "rarity": "SEC", "kind": "parallel"},
         "B": {"set": "OP06", "rarity": "SR", "kind": "base"}}


def truth(id, printing_id, **tags):
    return {id: {"printingId": printing_id, "cardId": printing_id.split("_")[0], "tags": tags}}


def test_summary_rates():
    predictions = [prediction("q1", "A", "B"), prediction("q2", "A", "A_p1"), prediction("q3", detected=False)]
    gt = {**truth("q1", "A"), **truth("q2", "A_p1"), **truth("q3", "B")}
    variants = {"A": "v1", "A_p1": "v1", "B": "v2"}
    result = metrics.summarize(predictions, gt, lambda p: ATTRS[p], variants.get)
    s = result["summary"]
    assert s["n"] == 3
    assert s["top1"] == round(1 / 3, 4)
    assert s["top3"] == round(2 / 3, 4)
    assert s["card_top1"] == round(2 / 3, 4)  # q2 predicted A: same card as A_p1
    assert s["variant_top1"] == 1.0  # q2 still spawns v1; q3 had no prediction (excluded)
    assert s["detection"] == round(2 / 3, 4)


def test_groups_have_precision_for_printing_attributes_only():
    predictions = [prediction("q1", "A"), prediction("q2", "A")]
    gt = {**truth("q1", "A", condition="glare"), **truth("q2", "B", condition="clean")}
    groups = metrics.summarize(predictions, gt, lambda p: ATTRS[p])["groups"]
    assert groups["set"]["OP05"] == {"n": 1, "recall": 1.0, "precision": 0.5}
    assert groups["set"]["OP06"]["recall"] == 0.0 and groups["set"]["OP06"]["precision"] is None
    assert "precision" not in groups["condition"]["glare"]


def test_hard_cases_most_confident_first():
    predictions = [prediction("q1", "B"), {**prediction("q2", "B"), "candidates": [
        {"printingId": "B", "cardId": "B", "similarity": 0.99, "matchesOCR": False}]}]
    gt = {**truth("q1", "A"), **truth("q2", "A")}
    cases = metrics.summarize(predictions, gt, lambda p: ATTRS[p])["hard_cases"]
    assert [c["id"] for c in cases] == ["q2", "q1"]


def test_rejection_curve_and_threshold():
    rows = [{"top1": True, "similarity": 0.9}, {"top1": True, "similarity": 0.7}, {"top1": False, "similarity": 0.95}]
    negatives = [{"candidates": [{"similarity": s}]} for s in (0.6, 0.65, 0.8)] + [{"candidates": []}]
    curve = metrics.rejection_curve(rows, negatives, thresholds=(0.6, 0.75, 0.85))
    assert curve[0] == {"threshold": 0.6, "correct_kept": round(2 / 3, 4), "false_accept": 0.75}
    assert curve[1] == {"threshold": 0.75, "correct_kept": round(1 / 3, 4), "false_accept": 0.25}
    assert curve[2]["false_accept"] == 0.0
    assert metrics.suggest_threshold(curve, max_false_accept=0.25) == 0.75
