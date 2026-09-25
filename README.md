# One Piece Card Battle AR

Put a physical One Piece TCG card on the desk and its character comes alive on top of it,
in the card's specific form. A Gear 4 Luffy card spawns Gear 4 Luffy, and a Gear 5 printing
spawns Gear 5. Fully on-device, iPhone only.

Personal project. Characters and card art are Bandai / Shueisha IP. Assets are gitignored and
never distributed.

## Layout

```
apps/ios/            SwiftUI + ARKit + RealityKit + Vision app, plus OnePieceKit (pure Swift package)
data/cards/          cards.json, printings.json, variants.json (bundled into the app at build time)
ml/                  offline Python lab: data fetch, embeddings, evaluation, Core ML export
docs/                architecture.md, cv-pipeline.md, asset-pipeline.md
```

## Getting started

Requirements: Xcode 27, and an iPhone XS or newer running iOS 18+. The simulator has no ARKit,
but the card browser and settings still run there.

```sh
make test     # OnePieceKit + BattleKit unit tests, no device needed
make build    # compile the app for a generic iPhone, unsigned
make open     # open in Xcode
```

To run on your phone: open the project, go to Signing & Capabilities for the OnePieceAR target,
choose your team (and change the bundle ID if it's taken), then run on the device.

### First run without any assets

The app works with no 3D models at all. Each variant spawns a colored placeholder figure with
the same behavior (idle, attack, hit, victory, wander), so you can test AR before the asset
pipeline exists.

1. **Pick card** > Summon. With no card art bundled, tap a surface to place the character.
2. Add card art as `apps/ios/OnePieceAR/Resources/Cards/<printingId>.png`. The character then
   anchors to the physical card, and **Scan card** starts recognizing it.
3. Add a model as `Resources/Characters/<variant>.usdz` plus clips (see
   `docs/asset-pipeline.md`), and the placeholder is replaced by the rigged character.

## Status

M0–M4 and M6 are implemented. The device half of M5 (scan logging) is done, and the Python side
is stubbed. The seed card data comes from the spec's example values and still needs to be
checked. See `docs/architecture.md` for details, and `data/cards/README.md`.
