# Card data

| File | Edited by | One entry per |
| --- | --- | --- |
| `roster.json` | **you** | card in the app: its character, default variant, per-printing variant overrides |
| `variants.json` | **you** | character form (`luffy_gear5`): model file, height, animations |
| `cards.json` | `fetch_cards.py` | card number (`OP05-119`), TCG stats from the OPTCG API |
| `printings.json` | `fetch_cards.py` | physical print (`OP05-119_p1`), matching the API's image IDs |
| `printings.f32` + `printings.meta.json` | `generate_embeddings.py` (gitignored) | reference embedding rows |

To change the roster, edit `roster.json` and run `uv run ml/scripts/fetch_cards.py`, then regenerate
embeddings. The app bundles everything here at build time. `swift test` validates the JSON
(`RepoDataTests`).

Variant resolution: the printing's override in `roster.json > printingVariants` wins, otherwise the
card's `defaultVariantId` applies.

**Roster status:** OP05-119 (Gear 5 Luffy, confirmed from the art of every printing) and OP06-118
(Zoro). The spec's third card, a Gear 4 Luffy, is still to pick. `luffy_gear4` exists in
variants.json but no card uses it yet.

## Index metadata

`printings.meta.json`: `backend` (must equal the device's embedding backend, or the app ignores the
index), `dimension`, `rows` (the printing ID of each row, and a printing may have several), and
`minimumSimilarity` (the device's rejection threshold, chosen from `evaluate.py`).
