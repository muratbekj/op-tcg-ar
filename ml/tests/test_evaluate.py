import csv

from oplab import dataset, evaluate, paths


def test_split_negatives_turns_indexed_cards_into_positives():
    negatives = [
        {"id": "negative:OP01-077/000_clean.jpg", "path": "x.jpg", "printingId": dataset.NEGATIVE, "cardId": dataset.NEGATIVE,
         "mode": "photo", "tags": {"source": "negative", "condition": "clean", "actual": "OP01-077"}},
        {"id": "negative:OP02-001/000_glare.jpg", "path": "y.jpg", "printingId": dataset.NEGATIVE, "cardId": dataset.NEGATIVE,
         "mode": "photo", "tags": {"source": "negative", "condition": "glare", "actual": "OP02-001"}},
    ]
    positives, true_negatives = evaluate.split_negatives(negatives, indexed={"OP01-077"})
    assert [p["printingId"] for p in positives] == ["OP01-077"]
    assert positives[0]["cardId"] == "OP01-077" and positives[0]["tags"]["source"] == "catalog"
    assert [n["id"] for n in true_negatives] == ["negative:OP02-001/000_glare.jpg"]


def test_append_result_migrates_old_header(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    old_columns = [c for c in evaluate.RESULT_COLUMNS if c not in ("ocr_accuracy", "within_group")]
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({"timestamp": "20260925-132157", "name": "featureprint-baseline", "top1": "0.7704"})
    monkeypatch.setattr(paths, "RESULTS", results)

    evaluate.append_result({"timestamp": "20260929-120000", "name": "code-first", "top1": 0.9, "within_group": 0.8})

    with results.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == evaluate.RESULT_COLUMNS
    assert [r["name"] for r in rows] == ["featureprint-baseline", "code-first"]
    assert rows[0]["top1"] == "0.7704" and rows[0]["within_group"] == ""
    assert rows[1]["within_group"] == "0.8"
