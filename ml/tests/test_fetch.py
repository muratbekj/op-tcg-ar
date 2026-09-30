from oplab import paths
from oplab.fetch import catalog_entries


def row(image_id, card_id, name="Luffy", **extra):
    return {"card_image_id": image_id, "card_set_id": card_id, "card_name": name, "set_id": "OP-05",
            "rarity": "SEC", "card_image": f"https://x/{image_id}.jpg", **extra}


def test_catalog_entries_sorted_and_deduplicated():
    rows = [row("OP05-119_p1", "OP05-119", "Monkey.D.Luffy (119) (Alternate Art)"),
            row("OP05-119", "OP05-119", "Monkey.D.Luffy (119)"),
            row("OP05-119", "OP05-119", "duplicate row")]
    entries = catalog_entries(rows)
    assert [e["printingId"] for e in entries] == ["OP05-119", "OP05-119_p1"]
    assert entries[0] == {"printingId": "OP05-119", "cardId": "OP05-119", "name": "Monkey.D.Luffy",
                          "set": "OP-05", "kind": "base", "rarity": "SEC", "artUrl": "https://x/OP05-119.jpg"}
    assert entries[1]["kind"] == "parallel"


def test_catalog_entries_tolerate_missing_rarity_and_art():
    entries = catalog_entries([row("P-001", "P-001", rarity=None, card_image=None)])
    assert entries[0]["rarity"] == "" and entries[0]["artUrl"] is None


def test_full_catalog_lives_with_card_data():
    assert paths.FULL_CATALOG == paths.DATA_CARDS / "catalog.json"
