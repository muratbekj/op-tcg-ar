from oplab import optcg
from oplab.fetch import build_roster_data


def row(image_id, name, card_id="OP05-119", **extra):
    return {"card_image_id": image_id, "card_set_id": card_id, "card_name": name, "card_type": "Character",
            "card_color": "Purple", "card_cost": "10", "card_power": "12000", "counter_amount": 0, "life": None,
            "rarity": "SEC", "set_id": "OP-05", "card_image": f"https://x/{image_id}.jpg", **extra}


def test_base_name_strips_qualifiers():
    assert optcg.base_name("Monkey.D.Luffy (119) (Alternate Art) (Manga)") == "Monkey.D.Luffy"
    assert optcg.base_name("Roronoa Zoro - OP06-118 (Reprint)") == "Roronoa Zoro"


def test_printing_kind():
    assert optcg.printing_kind(row("OP05-119", "Monkey.D.Luffy (119)")) == "base"
    assert optcg.printing_kind(row("OP05-119_p1", "Monkey.D.Luffy (119) (Alternate Art)")) == "parallel"
    assert optcg.printing_kind(row("OP05-119_p2", "Monkey.D.Luffy (119) (Alternate Art) (Manga)")) == "manga"
    assert optcg.printing_kind(row("OP05-119_r1", "Monkey.D.Luffy (OP05-119) (Reprint)")) == "base"


def test_to_card_parses_numbers_and_colors():
    card = optcg.to_card(row("OP05-119", "Monkey.D.Luffy (119)", card_color="Red Green"), "luffy", "luffy_gear5")
    assert card["set"] == "OP05" and card["number"] == "119"
    assert card["cost"] == 10 and card["power"] == 12000 and card["counter"] == 0
    assert card["colors"] == ["red", "green"]
    assert card["name"] == "Monkey.D.Luffy"


def test_dedupe_keeps_first():
    rows = [row("A", "a"), row("A", "dup"), row("B", "b")]
    assert [r["card_name"] for r in optcg.dedupe(rows)] == ["a", "b"]


def test_split_promo_id():
    assert optcg.split_card_id("P-001") == ("P", "001")


def test_build_roster_applies_overrides_and_keeps_rows():
    rows = [row("OP05-119", "Luffy (119)"), row("OP05-119_p1", "Luffy (119) (Alternate Art)")]
    roster = {"cards": [{"id": "OP05-119", "characterId": "luffy", "defaultVariantId": "luffy_gear5",
                         "printingVariants": {"OP05-119_p1": "luffy_alt", "OP05-119_p9": "x"}}]}
    cards, printings, warnings = build_roster_data(rows, roster, {"OP05-119": 3})
    assert [c["id"] for c in cards] == ["OP05-119"]
    assert {p["id"]: p["variantId"] for p in printings} == {"OP05-119": None, "OP05-119_p1": "luffy_alt"}
    assert printings[0]["embeddingRow"] == 3
    assert warnings == ["OP05-119: override for unknown printing OP05-119_p9"]
