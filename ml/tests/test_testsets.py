import pytest

from oplab import io, testsets


def record(scan_id, printing, split="test", label="confirmed"):
    return {"id": f"scan:{scan_id}", "scanId": scan_id, "path": f"/x/{scan_id}/crop.jpg", "printingId": printing,
            "label": label, "split": split}


def test_pool_is_labeled_test_scans_not_yet_frozen():
    records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P3")]
    sets = [{"name": "test-v1", "version": 1, "scans": [{"scanId": "a", "printingId": "P1"}], "printings": ["P1"]}]
    assert [r["scanId"] for r in testsets.pool(records, sets)] == ["c"]


def test_freeze_refuses_below_thresholds_and_reports_progress(tmp_path):
    records = [record(f"s{i}", f"P{i % 3}") for i in range(10)]
    with pytest.raises(testsets.FreezeError, match="10/200 labeled scans across 3/30 printings"):
        testsets.freeze(records, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_freeze_writes_versioned_set(tmp_path):
    records = [record(f"s{i:02d}", f"P{i % 4}") for i in range(8)] + [record("t1", "P9", split="train")]
    frozen = testsets.freeze(records, tmp_path, min_scans=8, min_printings=4, today="2026-10-01")
    assert frozen["name"] == "test-v1" and frozen["version"] == 1 and frozen["frozen"] == "2026-10-01"
    assert [s["scanId"] for s in frozen["scans"]] == [f"s{i:02d}" for i in range(8)]
    assert frozen["printings"] == ["P0", "P1", "P2", "P3"]
    assert io.read_json(tmp_path / "test-v1.json") == frozen
    assert testsets.load("latest", tmp_path) == frozen == testsets.load("test-v1", tmp_path)


def test_second_freeze_uses_only_new_scans(tmp_path):
    first = [record(f"a{i}", f"P{i % 2}") for i in range(4)]
    testsets.freeze(first, tmp_path, min_scans=4, min_printings=2, today="2026-10-01")
    before = (tmp_path / "test-v1.json").read_text()
    later = first + [record(f"b{i}", f"P{i % 3}") for i in range(3)]
    second = testsets.freeze(later, tmp_path, min_scans=3, min_printings=3, today="2026-11-01")
    assert second["name"] == "test-v2" and [s["scanId"] for s in second["scans"]] == ["b0", "b1", "b2"]
    assert (tmp_path / "test-v1.json").read_text() == before
    assert [s["name"] for s in testsets.frozen_sets(tmp_path)] == ["test-v1", "test-v2"]
    assert testsets.load("latest", tmp_path)["name"] == "test-v2"


def test_relabeled_frozen_scan_keeps_frozen_truth(tmp_path):
    testsets.freeze([record("a", "P1"), record("b", "P2")], tmp_path, min_scans=2, min_printings=2, today="2026-10-01")
    # Later the user corrects scan "a" to another test printing; it must not re-enter the pool.
    relabeled = [record("a", "P7", label="corrected"), record("b", "P2")]
    assert testsets.pool(relabeled, testsets.frozen_sets(tmp_path)) == []
    assert testsets.load("test-v1", tmp_path)["scans"][0] == {"scanId": "a", "printingId": "P1"}


def test_load_missing_set_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        testsets.load("latest", tmp_path)
    with pytest.raises(FileNotFoundError):
        testsets.load("test-v3", tmp_path)


def test_status_lines():
    all_records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P1", label="none"),
                   record("d", "P3", label="corrected")]
    sets = [{"name": "test-v1", "version": 1, "frozen": "2026-10-01", "scans": [{"scanId": "d", "printingId": "P3"}],
             "printings": ["P3"]}]
    lines = testsets.status_lines(all_records, sets, min_scans=200, min_printings=30)
    assert lines == [
        "scans: 4 imported, 3 labeled (2 confirmed, 1 corrected), 1 unlabeled",
        "labeled split: 1 train, 2 test",
        "test pool for test-v2: 1/200 scans across 1/30 printings",
        "frozen: test-v1 (2026-10-01, 1 scans, 1 printings)",
    ]
