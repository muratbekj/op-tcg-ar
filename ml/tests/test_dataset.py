import json

from oplab import dataset


def write_scan(root, scan_id, final="OP01-003", label=None, corrected=False, crop=True):
    folder = root / scan_id
    folder.mkdir(parents=True)
    record = {"id": scan_id, "date": "2026-09-30T10:00:00Z", "candidates": [], "spawnedPrintingID": final,
              "finalPrintingID": final, "corrected": corrected, "ocrCardID": None}
    if label is not None:
        record["label"] = label
    (folder / "scan.json").write_text(json.dumps(record))
    if crop:
        (folder / "crop.jpg").write_bytes(b"jpeg")
    return folder


def test_printing_split_is_stable_and_roughly_30_percent():
    ids = [f"OP{i // 1000:02d}-{i % 1000:03d}" for i in range(3000)]
    splits = [dataset.printing_split(i) for i in ids]
    assert splits == [dataset.printing_split(i) for i in ids]
    assert set(splits) == {"train", "test"}
    assert 0.25 < splits.count("test") / len(splits) < 0.35
    assert dataset.printing_split("OP01-003") == "test" and dataset.printing_split("OP05-119") == "train"


def test_scan_label_reads_label_and_falls_back_for_old_records():
    assert dataset.scan_label({"label": "confirmed", "corrected": False}) == "confirmed"
    assert dataset.scan_label({"label": "none", "corrected": False}) == "none"
    assert dataset.scan_label({"corrected": True}) == "corrected"   # logged before labels existed
    assert dataset.scan_label({"corrected": False}) == "none"        # untouched is NOT a weak label anymore


def test_scan_records_keep_only_labeled_scans_and_split_by_printing(tmp_path):
    write_scan(tmp_path, "s1", final="OP01-003", label="confirmed")
    write_scan(tmp_path, "s2", final="OP05-119", label="corrected", corrected=True)
    write_scan(tmp_path, "s3", final="OP01-003", label="none")
    write_scan(tmp_path, "s4", final="OP05-119", corrected=True)      # old format
    records = dataset.scan_records(tmp_path)
    assert [(r["scanId"], r["label"], r["split"]) for r in records] == [
        ("s1", "confirmed", "test"), ("s2", "corrected", "train"), ("s4", "corrected", "train")]
    assert records[0]["id"] == "scan:s1" and records[0]["printingId"] == "OP01-003"
    assert records[0]["path"].endswith("s1/crop.jpg")
    assert len(dataset.scan_records(tmp_path, labeled_only=False)) == 4


def test_scan_records_skip_broken_records(tmp_path):
    write_scan(tmp_path, "ok", label="confirmed")
    write_scan(tmp_path, "nocrop", label="confirmed", crop=False)
    (tmp_path / "badjson").mkdir()
    (tmp_path / "badjson" / "scan.json").write_text("{not json")
    (tmp_path / "badjson" / "crop.jpg").write_bytes(b"jpeg")
    (tmp_path / "nofinal").mkdir()
    (tmp_path / "nofinal" / "scan.json").write_text(json.dumps({"id": "nofinal", "label": "confirmed"}))
    (tmp_path / "nofinal" / "crop.jpg").write_bytes(b"jpeg")
    assert [r["scanId"] for r in dataset.scan_records(tmp_path)] == ["ok"]


def test_import_refreshes_changed_labels(tmp_path):
    phone, lab = tmp_path / "phone", tmp_path / "lab"
    write_scan(phone, "s1", label="none")
    assert dataset.import_scans(phone, lab) == {"new": 1, "updated": 0, "skipped": []}
    # The user confirms the scan on the phone later and exports again.
    record = json.loads((phone / "s1" / "scan.json").read_text())
    record["label"] = "confirmed"
    (phone / "s1" / "scan.json").write_text(json.dumps(record))
    assert dataset.import_scans(phone, lab) == {"new": 0, "updated": 1, "skipped": []}
    assert json.loads((lab / "s1" / "scan.json").read_text())["label"] == "confirmed"
    assert dataset.import_scans(phone, lab) == {"new": 0, "updated": 0, "skipped": []}


def test_import_skips_malformed_folders(tmp_path):
    phone, lab = tmp_path / "phone", tmp_path / "lab"
    write_scan(phone, "good", label="confirmed")
    write_scan(phone, "nocrop", label="confirmed", crop=False)
    result = dataset.import_scans(phone, lab)
    assert result == {"new": 1, "updated": 0, "skipped": ["nocrop"]}
    assert (phone / "nocrop" / "scan.json").exists()                  # left in place
    assert not (lab / "nocrop").exists()
