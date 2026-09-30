from oplab import dataset, embeddings, io, paths


def test_full_references_merge_roster_and_catalog(tmp_path, monkeypatch):
    art, app_cards, data = tmp_path / "art", tmp_path / "app", tmp_path / "data"
    for d in (art, app_cards, data):
        d.mkdir()
    (art / "OP05-119.jpg").write_bytes(b"roster-api")
    (art / "OP01-077.jpg").write_bytes(b"catalog-only")
    (art / "OP01-078.jpg").write_bytes(b"roster-api")          # same bytes as a roster image: deduplicated
    (app_cards / "OP05-119.png").write_bytes(b"own-clean-scan")
    io.write_json(data / "printings.json", [{"id": "OP05-119"}])
    io.write_json(data / "catalog.json", [{"printingId": p} for p in ("OP01-077", "OP01-078", "OP05-119", "OP09-999")])
    monkeypatch.setattr(paths, "ART", art)
    monkeypatch.setattr(paths, "APP_CARDS", app_cards)
    monkeypatch.setattr(paths, "DATA_CARDS", data)
    monkeypatch.setattr(paths, "FULL_CATALOG", data / "catalog.json")

    entries = embeddings.full_references()
    assert [(e["printingId"], e["source"]) for e in entries] == [
        ("OP05-119", "api"), ("OP05-119", "app"), ("OP01-077", "api")]   # OP09-999 has no art


def test_scan_references_respect_split_label_scope_and_dedup(tmp_path, monkeypatch):
    def record(name, printing, label, split):
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(name.encode())
        return {"id": f"scan:{name}", "scanId": name, "path": str(path), "printingId": printing,
                "label": label, "split": split}

    records = [
        record("held_out", "OP01-003", "confirmed", "test"),
        record("confirmed", "OP05-119", "confirmed", "train"),
        record("corrected", "OP05-119", "corrected", "train"),
        record("elsewhere", "OP09-001", "corrected", "train"),
    ]
    monkeypatch.setattr(dataset, "train_records", lambda *a, **k: [r for r in records if r["split"] == "train"])

    def ids(with_scans, printing_ids, seen=None):
        return [e["path"].rsplit("/", 1)[1][:-4] for e in
                embeddings.scan_references(with_scans, printing_ids, set() if seen is None else seen)]

    assert ids("none", None) == []
    assert ids("labeled", None) == ["confirmed", "corrected", "elsewhere"]      # test split never included
    assert ids("corrected", None) == ["corrected", "elsewhere"]
    assert ids("labeled", {"OP05-119"}) == ["confirmed", "corrected"]           # roster scope filters
    seen = {embeddings._digest(tmp_path / "confirmed.jpg")}
    assert ids("labeled", None, seen) == ["corrected", "elsewhere"]             # already embedded
    assert len(seen) == 3                                                       # and new digests are recorded
