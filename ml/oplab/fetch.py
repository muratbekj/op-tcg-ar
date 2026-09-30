"""Build data/cards/{catalog,cards,printings}.json from roster.json + the OPTCG API, and download art.

roster.json is the hand-edited source of truth: which cards are in the app, their character, their
default variant, and per-printing variant overrides. cards.json and printings.json are regenerated
from it; variants.json is never touched.
"""

import argparse
import shutil

from . import io, optcg, paths


def build_roster_data(rows: list[dict], roster: dict, previous_rows: dict[str, int | None]) -> tuple[list, list, list[str]]:
    """Returns (cards, printings, warnings) for the roster cards."""
    by_card: dict[str, list[dict]] = {}
    for row in rows:
        by_card.setdefault(row["card_set_id"], []).append(row)

    cards, printings, warnings = [], [], []
    for entry in roster["cards"]:
        card_rows = sorted(by_card.get(entry["id"], []), key=lambda r: r["card_image_id"])
        if not card_rows:
            warnings.append(f"{entry['id']}: not found in the API")
            continue
        base = next((r for r in card_rows if r["card_image_id"] == entry["id"]), card_rows[0])
        cards.append(optcg.to_card(base, entry["characterId"], entry["defaultVariantId"]))

        overrides = entry.get("printingVariants", {})
        known = {r["card_image_id"] for r in card_rows}
        for printing_id in overrides.keys() - known:
            warnings.append(f"{entry['id']}: override for unknown printing {printing_id}")
        for row in card_rows:
            printing_id = row["card_image_id"]
            printings.append(optcg.to_printing(row, overrides.get(printing_id), previous_rows.get(printing_id)))
    return cards, printings, warnings


def catalog_entries(rows: list[dict]) -> list[dict]:
    """Every printing in the API, one entry each, sorted by printing ID so the file diffs cleanly."""
    entries: dict[str, dict] = {}
    for r in rows:
        entries.setdefault(r["card_image_id"], {
            "printingId": r["card_image_id"], "cardId": r["card_set_id"], "name": optcg.base_name(r["card_name"]),
            "set": r["set_id"], "kind": optcg.printing_kind(r), "rarity": r.get("rarity") or "",
            "artUrl": r.get("card_image"),
        })
    return [entries[key] for key in sorted(entries)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download API data instead of using the cache")
    parser.add_argument("--art", choices=["roster", "all", "none"], default="roster",
                        help="which card art to download into data/cards/art (all is ~4k images)")
    parser.add_argument("--install-art", action="store_true",
                        help="copy roster art into the app's Resources/Cards, never overwriting existing files")
    args = parser.parse_args(argv)

    print("Fetching OPTCG API…")
    rows = optcg.fetch_rows(paths.API_CACHE, refresh=args.refresh)
    print(f"  {len(rows)} printings")

    catalog = catalog_entries(rows)
    io.write_json(paths.FULL_CATALOG, catalog)
    print(f"  catalog: {len(catalog)} printings -> {paths.FULL_CATALOG.relative_to(paths.REPO)}")

    roster = io.read_json(paths.ROSTER)
    previous_path = paths.DATA_CARDS / "printings.json"
    previous = {p["id"]: p.get("embeddingRow") for p in io.read_json(previous_path)} if previous_path.exists() else {}
    cards, printings, warnings = build_roster_data(rows, roster, previous)

    variant_ids = {v["id"] for v in io.read_json(paths.DATA_CARDS / "variants.json")}
    for card in cards:
        if card["defaultVariantId"] not in variant_ids:
            warnings.append(f"{card['id']}: default variant {card['defaultVariantId']} is not in variants.json")

    io.write_json(paths.DATA_CARDS / "cards.json", cards)
    io.write_json(previous_path, printings)
    print(f"  wrote {len(cards)} cards, {len(printings)} printings")
    for warning in warnings:
        print(f"  warning: {warning}")

    roster_ids = {p["id"] for p in printings}
    if args.art != "none":
        wanted = rows if args.art == "all" else [r for r in rows if r["card_image_id"] in roster_ids]
        downloaded, present, failed = optcg.download_art(wanted, paths.ART)
        print(f"  art: {downloaded} downloaded, {present} already present, {len(failed)} failed"
              f"{' ' + ', '.join(failed[:5]) if failed else ''} -> {paths.ART.relative_to(paths.REPO)}/")

    installed = [p for p in sorted(roster_ids) if any(paths.APP_CARDS.glob(f"{p}.*"))]
    if args.install_art:
        paths.APP_CARDS.mkdir(parents=True, exist_ok=True)
        for printing_id in sorted(roster_ids):
            source = paths.ART / f"{printing_id}.jpg"
            if source.exists() and printing_id not in installed:
                shutil.copy(source, paths.APP_CARDS / source.name)
                installed.append(printing_id)
        print(f"  app art: {len(installed)}/{len(roster_ids)} roster printings in "
              f"{paths.APP_CARDS.relative_to(paths.REPO)}/")
    elif len(installed) < len(roster_ids):
        print(f"  app art: {len(installed)}/{len(roster_ids)} roster printings bundled in the app; "
              f"add --install-art to copy the rest (needed for card tracking in AR)")


if __name__ == "__main__":
    main()
