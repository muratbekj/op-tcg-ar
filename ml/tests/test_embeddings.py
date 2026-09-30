from oplab import embeddings, io, paths


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
