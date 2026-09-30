"""OPTCG API client and mapping from API rows to the app's Card / Printing schema.

One API row is one printing. `card_image_id` ("OP05-119_p1") is the printing ID and matches the
art file name; `card_set_id` ("OP05-119") is the card number shared by every printing.
"""

import re
import time
from pathlib import Path

import requests

from . import io

API = "https://optcgapi.com/api"
ENDPOINTS = ["allSetCards", "allSTCards", "allPromoCards"]
COLORS = {"red", "green", "blue", "purple", "black", "yellow"}
_PAREN = re.compile(r"\s*\(([^)]*)\)")
_DASH_SUFFIX = re.compile(r"\s+-\s+[A-Z]+\d*-\d{3}.*$")


def fetch_rows(cache_dir: Path, refresh: bool = False) -> list[dict]:
    """All printings from every endpoint, cached as raw JSON. Endpoints that 404 are skipped."""
    rows: list[dict] = []
    for endpoint in ENDPOINTS:
        cache = cache_dir / f"{endpoint}.json"
        if refresh or not cache.exists():
            response = requests.get(f"{API}/{endpoint}/", timeout=120)
            if response.status_code == 404:
                print(f"  {endpoint}: not available (404), skipping")
                continue
            response.raise_for_status()
            io.write_json(cache, response.json())
        endpoint_rows = io.read_json(cache)
        for row in endpoint_rows:
            row["_endpoint"] = endpoint
        rows += endpoint_rows
    return dedupe(rows)


def dedupe(rows: list[dict]) -> list[dict]:
    """The API lists some printings twice; keep the first of each `card_image_id`."""
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["card_image_id"] not in seen:
            seen.add(row["card_image_id"])
            unique.append(row)
    return unique


def _int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).replace(",", ""))
    except ValueError:
        return None


def name_tags(card_name: str) -> list[str]:
    """Parenthesized qualifiers, e.g. "Monkey.D.Luffy (119) (Alternate Art)" -> ["119", "Alternate Art"]."""
    return _PAREN.findall(card_name)


def base_name(card_name: str) -> str:
    """"Roronoa Zoro - OP06-118 (Reprint)" -> "Roronoa Zoro"."""
    return _DASH_SUFFIX.sub("", _PAREN.sub("", card_name)).strip()


def split_card_id(card_id: str) -> tuple[str, str]:
    """"OP05-119" -> ("OP05", "119"); "P-001" -> ("P", "001")."""
    set_code, _, number = card_id.partition("-")
    return set_code, number


def printing_kind(row: dict) -> str:
    tags = name_tags(row["card_name"])
    if "Manga" in tags:
        return "manga"
    if row.get("_endpoint") == "allPromoCards":
        return "promo"
    # Plain reprints carry the base art.
    if row["card_image_id"] == row["card_set_id"] or "Reprint" in tags:
        return "base"
    return "parallel"


def to_card(row: dict, character_id: str, default_variant_id: str) -> dict:
    set_code, number = split_card_id(row["card_set_id"])
    kind = row["card_type"].lower()
    return {
        "id": row["card_set_id"],
        "set": set_code,
        "number": number,
        "name": base_name(row["card_name"]),
        "kind": kind if kind in {"leader", "character", "event", "stage"} else "character",
        "colors": [c for c in row["card_color"].lower().split() if c in COLORS],
        "cost": _int(row.get("card_cost")),
        "power": _int(row.get("card_power")),
        "counter": _int(row.get("counter_amount")),
        "life": _int(row.get("life")),
        "characterId": character_id,
        "defaultVariantId": default_variant_id,
    }


def to_printing(row: dict, variant_id: str | None = None, embedding_row: int | None = None) -> dict:
    return {
        "id": row["card_image_id"],
        "cardId": row["card_set_id"],
        "kind": printing_kind(row),
        "rarity": row.get("rarity") or "",
        "language": "en",
        "artUrl": row.get("card_image"),
        "artCrop": None,
        "embeddingRow": embedding_row,
        "variantId": variant_id,
    }


def download_art(rows: list[dict], art_dir: Path, delay: float = 0.05) -> tuple[int, int, list[str]]:
    """Downloads missing art as <printingId>.jpg. Returns (downloaded, already present, failed IDs)."""
    art_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    downloaded, present, failed = 0, 0, []
    for index, row in enumerate(rows, 1):
        target = art_dir / f"{row['card_image_id']}.jpg"
        if target.exists():
            present += 1
            continue
        if not row.get("card_image"):
            failed.append(row["card_image_id"])
            continue
        for attempt in range(3):
            try:
                response = session.get(row["card_image"], timeout=60)
                response.raise_for_status()
                target.write_bytes(response.content)
                downloaded += 1
                break
            except requests.RequestException:
                time.sleep(1 + attempt)
        else:
            failed.append(row["card_image_id"])
        if index % 200 == 0:
            print(f"  art {index}/{len(rows)}")
        time.sleep(delay)
    return downloaded, present, failed
