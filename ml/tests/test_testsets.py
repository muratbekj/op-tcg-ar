import pytest

from oplab import io, testsets


ALL = {f"P{i}" for i in range(10)}


def record(scan_id, printing, split="test", label="confirmed", method=None):
    return {"method": method, "id": f"scan:{scan_id}", "scanId": scan_id, "path": f"/x/{scan_id}/crop.jpg", "printingId": printing,
            "label": label, "split": split}


def test_pool_is_labeled_test_scans_not_yet_frozen():
    records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P3")]
    sets = [{"name": "test-v1", "version": 1, "scans": [{"scanId": "a", "printingId": "P1"}], "printings": ["P1"]}]
    assert [r["scanId"] for r in testsets.pool(records, sets)] == ["c"]


def test_freeze_refuses_below_thresholds_and_reports_progress(tmp_path):
    records = [record(f"s{i}", f"P{i % 3}") for i in range(10)]
    with pytest.raises(testsets.FreezeError, match="10/200 labeled scans across 3/30 printings"):
        testsets.freeze(records, tmp_path, indexed=ALL)
    assert list(tmp_path.iterdir()) == []


def test_freeze_writes_versioned_set(tmp_path):
    records = [record(f"s{i:02d}", f"P{i % 4}") for i in range(8)] + [record("t1", "P9", split="train")]
    frozen = testsets.freeze(records, tmp_path, indexed=ALL, min_scans=8, min_printings=4, today="2026-10-01")
    assert frozen["name"] == "test-v1" and frozen["version"] == 1 and frozen["frozen"] == "2026-10-01"
    assert [s["scanId"] for s in frozen["scans"]] == [f"s{i:02d}" for i in range(8)]
    assert frozen["printings"] == ["P0", "P1", "P2", "P3"]
    assert io.read_json(tmp_path / "test-v1.json") == frozen
    assert testsets.load("latest", tmp_path) == frozen == testsets.load("test-v1", tmp_path)


def test_second_freeze_uses_only_new_scans(tmp_path):
    first = [record(f"a{i}", f"P{i % 2}") for i in range(4)]
    testsets.freeze(first, tmp_path, indexed=ALL, min_scans=4, min_printings=2, today="2026-10-01")
    before = (tmp_path / "test-v1.json").read_text()
    later = first + [record(f"b{i}", f"P{i % 3}") for i in range(3)]
    second = testsets.freeze(later, tmp_path, indexed=ALL, min_scans=3, min_printings=3, today="2026-11-01")
    assert second["name"] == "test-v2" and [s["scanId"] for s in second["scans"]] == ["b0", "b1", "b2"]
    assert (tmp_path / "test-v1.json").read_text() == before
    assert [s["name"] for s in testsets.frozen_sets(tmp_path)] == ["test-v1", "test-v2"]
    assert testsets.load("latest", tmp_path)["name"] == "test-v2"


def test_relabeled_frozen_scan_keeps_frozen_truth(tmp_path):
    testsets.freeze([record("a", "P1"), record("b", "P2")], tmp_path, indexed=ALL, min_scans=2, min_printings=2, today="2026-10-01")
    # Later the user corrects scan "a" to another test printing; it must not re-enter the pool.
    relabeled = [record("a", "P7", label="corrected"), record("b", "P2")]
    assert testsets.pool(relabeled, testsets.frozen_sets(tmp_path)) == []
    assert testsets.load("test-v1", tmp_path)["scans"][0] == {"scanId": "a", "printingId": "P1", "label": "confirmed",
                                                                  "method": None}


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


FROZEN = {"name": "test-v1", "version": 1, "frozen": "2026-10-01",
          "scans": [{"scanId": "a", "printingId": "OP09-078-r1", "label": "corrected", "method": "ocr"},
                    {"scanId": "b", "printingId": "OP01-003", "label": "confirmed", "method": None}],
          "printings": ["OP01-003", "OP09-078-r1"]}


def make_crops(root, *scan_ids):
    for scan_id in scan_ids:
        (root / scan_id).mkdir(parents=True)
        (root / scan_id / "crop.jpg").write_bytes(b"jpeg")


def test_testset_entries_use_frozen_truth(tmp_path):
    make_crops(tmp_path, "a", "b")
    rows = testsets.entries(FROZEN, tmp_path, card_ids={"OP09-078-r1": "OP09-078"})
    assert rows[0] == {"id": "scan:a", "path": str(tmp_path / "a" / "crop.jpg"), "printingId": "OP09-078-r1",
                       "cardId": "OP09-078", "mode": "card",
                       "tags": {"source": "scan", "testset": "test-v1", "label": "corrected"}}
    assert rows[1]["printingId"] == "OP01-003" and rows[1]["cardId"] == "OP01-003"


def test_testset_entries_fail_on_missing_crops(tmp_path):
    make_crops(tmp_path, "a")
    with pytest.raises(testsets.MissingScans, match="1 of 2 scans in test-v1 have no crop.*b"):
        testsets.entries(FROZEN, tmp_path)


def test_unindexed_printings_are_not_pooled_or_frozen(tmp_path):
    records = [record("a", "P1"), record("b", "P2"), record("c", "NOART")]
    assert [r["scanId"] for r in testsets.pool(records, [], {"P1", "P2"})] == ["a", "b"]
    frozen = testsets.freeze(records, tmp_path, indexed={"P1", "P2"}, min_scans=2, min_printings=2)
    assert [s["scanId"] for s in frozen["scans"]] == ["a", "b"] and frozen["printings"] == ["P1", "P2"]


def test_freeze_stores_label_and_method(tmp_path):
    records = [record("a", "P1", label="corrected", method="ocr"), record("b", "P2")]
    frozen = testsets.freeze(records, tmp_path, indexed=ALL, min_scans=2, min_printings=2)
    assert frozen["scans"] == [{"scanId": "a", "printingId": "P1", "label": "corrected", "method": "ocr"},
                               {"scanId": "b", "printingId": "P2", "label": "confirmed", "method": None}]


def test_frozen_sets_ignore_copies(tmp_path):
    for name in ("test-v1.json", "test-v1 copy.json", "test-v2 (1).json", "test-vx.json"):
        io.write_json(tmp_path / name, {"name": name, "version": 1, "scans": [], "printings": []})
    assert [s["name"] for s in testsets.frozen_sets(tmp_path)] == ["test-v1.json"]


def test_status_reports_unfreezable_and_relabeled():
    all_records = [record("a", "P1"), record("b", "NOART"), record("f", "P4", split="train", label="corrected")]
    sets = [{"name": "test-v1", "version": 1, "frozen": "2026-10-01", "printings": ["P3"],
             "scans": [{"scanId": "f", "printingId": "P3"}]}]
    lines = testsets.status_lines(all_records, sets, indexed={"P1", "P4"})
    assert "test pool for test-v2: 1/200 scans across 1/30 printings" in lines
    assert "not freezable (printing not in the index): 1 scans" in lines
    assert "frozen scans relabeled since freeze: 1" in lines
    assert not any("not freezable" in l for l in testsets.status_lines(all_records, sets))


def test_pool_set_scores_the_unfrozen_test_pool(tmp_path):
    records = [record("a", "P1"), record("b", "P2", split="train"), record("c", "P3", label="none"),
               record("d", "P4", method="ocr")]
    sets = [{"name": "test-v1", "version": 1, "scans": [{"scanId": "a", "printingId": "P1"}], "printings": ["P1"]}]
    pool_set = testsets.pool_set(records, sets)
    assert pool_set["name"] == testsets.POOL_NAME == "test-pool"
    assert pool_set["scans"] == [{"scanId": "d", "printingId": "P4", "label": "confirmed", "method": "ocr"}]
    make_crops(tmp_path, "d")
    rows = testsets.entries(pool_set, tmp_path, card_ids={})
    assert [r["id"] for r in rows] == ["scan:d"] and rows[0]["tags"]["testset"] == "test-pool"
