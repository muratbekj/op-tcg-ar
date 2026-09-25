"""Fetch card metadata and art from the OPTCG API (https://optcgapi.com/).

Writes data/cards/cards.json and printings.json in the schema documented in
data/cards/README.md, and downloads art to data/cards/art/<printingId>.<ext>.
Never overwrite hand-set `variantId` overrides in printings.json or variants.json.

Copy the art for roster printings into apps/ios/OnePieceAR/Resources/Cards/ (gitignored). The app
uses it for image tracking and for on-device reference embeddings.
"""

raise SystemExit("fetch_cards.py: not implemented yet")
