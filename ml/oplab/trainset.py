"""What a training epoch is made of, without torch so it can be tested anywhere.

Items: `views` augmented views of every catalog printing's art, plus each labeled train-split scan
repeated `scan_weight` times (real scans are scarce and closer to what the phone sees). Batches are
group-aware: printings that share a card code (base, alt art, manga, reprints) are placed in the same
batch, so the CosFace loss has to separate exactly the siblings the app must tell apart.
"""

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    path: str
    label: int
    is_scan: bool


def build_items(classes: list[str], art_paths: dict[str, str], scans: list[dict], views: int,
                scan_weight: int) -> tuple[list[Item], dict]:
    index = {printing: label for label, printing in enumerate(classes)}
    items = [Item(art_paths[printing], label, False) for label, printing in enumerate(classes) for _ in range(views)]
    usable = [s for s in scans if s["printingId"] in index]
    for record in usable:
        items += [Item(record["path"], index[record["printingId"]], True)] * scan_weight
    counts = {"printings": len(classes), "art_views": len(classes) * views, "scans": len(usable),
              "scan_printings": len({s["printingId"] for s in usable}), "scans_skipped": len(scans) - len(usable),
              "scan_weight": scan_weight}
    return items, counts


def group_batches(labels: list[int], codes: list[str], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Every item index exactly once. Items of a multi-printing code are dealt round-robin across its
    printings into consecutive slots, so siblings land in the same batch; single-printing items fill in."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    for indices in by_label.values():
        rng.shuffle(indices)
    by_code: dict[str, list[int]] = defaultdict(list)
    for label in by_label:
        by_code[codes[label]].append(label)

    chunks: list[list[int]] = []
    for group in by_code.values():
        if len(group) < 2:
            chunks += [[i] for label in group for i in by_label[label]]
            continue
        queues = [list(by_label[label]) for label in group]
        while any(queues):
            chunk = [q.pop() for q in queues if q]  # one item from each sibling printing
            chunks.append(chunk)
    rng.shuffle(chunks)

    batches, current = [], []
    for chunk in chunks:
        if current and len(current) + len(chunk) > batch_size:
            batches.append(current)
            current = []
        if len(chunk) > batch_size:  # a group wider than the batch: split it
            for start in range(0, len(chunk), batch_size):
                batches.append(chunk[start:start + batch_size])
            continue
        current += chunk
    if current:
        batches.append(current)
    return batches
