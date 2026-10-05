import random
from collections import Counter

from oplab import trainset


CLASSES = ["OP05-119", "OP05-119_p1", "OP06-118", "OP01-001"]
ART = {c: f"/art/{c}.jpg" for c in CLASSES}


def scan(scan_id, printing):
    return {"scanId": scan_id, "printingId": printing, "path": f"/scans/{scan_id}/crop.jpg"}


def test_items_combine_art_views_and_oversampled_scans():
    items, counts = trainset.build_items(CLASSES, ART, [scan("s1", "OP05-119_p1"), scan("s2", "OP06-118")],
                                         views=2, scan_weight=4)
    assert Counter(i.is_scan for i in items) == {False: 8, True: 8}
    assert sum(1 for i in items if i.path == "/scans/s1/crop.jpg") == 4
    assert {i.label for i in items if i.path == "/scans/s1/crop.jpg"} == {CLASSES.index("OP05-119_p1")}
    assert counts == {"printings": 4, "art_views": 8, "scans": 2, "scan_printings": 2, "scans_skipped": 0, "scan_weight": 4}


def test_items_without_scans():
    items, counts = trainset.build_items(CLASSES, ART, [], views=3, scan_weight=4)
    assert len(items) == 12 and not any(i.is_scan for i in items)
    assert counts["scans"] == 0 and counts["scan_printings"] == 0


def test_scans_without_a_class_are_skipped_and_counted():
    items, counts = trainset.build_items(CLASSES, ART, [scan("s1", "OP09-999"), scan("s2", "OP01-001")],
                                         views=1, scan_weight=2)
    assert counts["scans"] == 1 and counts["scans_skipped"] == 1
    assert all(i.path != "/scans/s1/crop.jpg" for i in items)


def test_group_batches_cover_every_item_once():
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 3]
    codes = ["OP05-119", "OP05-119", "OP06-118", "OP01-001"]
    for batch_size in (2, 4, 64):
        batches = trainset.group_batches(labels, codes, batch_size, random.Random(0))
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(labels)))
        assert all(len(b) <= batch_size for b in batches)
    # No multi-printing codes at all: still a plain shuffled cover.
    batches = trainset.group_batches([0, 1, 2], ["A", "B", "C"], 2, random.Random(1))
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_group_batches_keep_siblings_together():
    # Two printings of OP05-119 (classes 0, 1) among many single-printing classes.
    labels = [0] * 8 + [1] * 8 + list(range(2, 42))
    codes = ["OP05-119", "OP05-119"] + [f"OP01-{n:03d}" for n in range(40)]
    batches = trainset.group_batches(labels, codes, 8, random.Random(3))
    together = sum(1 for b in batches if {labels[i] for i in b} >= {0, 1})
    assert together >= 4   # most sibling items share a batch with the other printing
