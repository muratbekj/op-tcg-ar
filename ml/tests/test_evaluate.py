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


def test_split_negatives_takes_card_id_from_the_catalog():
    negatives = [{"id": "n", "path": "x.jpg", "printingId": dataset.NEGATIVE, "cardId": dataset.NEGATIVE, "mode": "photo",
                  "tags": {"source": "negative", "condition": "clean", "actual": "OP09-078-r1"}}]
    positives, _ = evaluate.split_negatives(negatives, {"OP09-078-r1"}, {"OP09-078-r1": "OP09-078"})
    assert positives[0]["cardId"] == "OP09-078"
    positives, _ = evaluate.split_negatives(negatives, {"OP09-078-r1"})
    assert positives[0]["cardId"] == "OP09-078-r1"  # no catalog: falls back to the split


def test_append_result_migrates_old_header(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    old_columns = evaluate.RESULT_COLUMNS[:evaluate.RESULT_COLUMNS.index("ocr_accuracy")]
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


def test_interval_columns_flatten_summary_intervals():
    summary = {"top1_ci": [0.1, 0.9], "ocr_accuracy_ci": [0.2, 0.8], "within_group_ci": [None, None]}
    columns = evaluate.interval_columns(summary)
    assert columns == {"top1_low": 0.1, "top1_high": 0.9, "ocr_accuracy_low": 0.2, "ocr_accuracy_high": 0.8,
                       "within_group_low": None, "within_group_high": None}
    assert set(columns) <= set(evaluate.RESULT_COLUMNS)


def test_testset_problems():
    entries = [{"printingId": "A"}, {"printingId": "B"}, {"printingId": "B"}]
    assert evaluate.testset_problems(entries, {"A", "B"}, None, None) is None
    assert "--limit" in evaluate.testset_problems(entries, {"A", "B"}, 5, None)
    assert "--source" in evaluate.testset_problems(entries, {"A", "B"}, None, ["photo"])
    message = evaluate.testset_problems(entries, {"A"}, None, None)
    assert "2 of 3" in message and "B" in message and "full catalog" in message


def test_scans_cant_be_combined_with_testset_limit_or_source():
    import pytest
    for extra in (["--testset", "latest"], ["--limit", "5"], ["--source", "photo"]):
        with pytest.raises(SystemExit, match="--scans test"):
            evaluate.main(["--scans", "test", *extra])
