# ML lab

Offline, on the Mac. The phone never talks to a server. The lab produces files the app bundles
(`data/cards/*`, optionally a Core ML model) and measures how well recognition works.

**The key idea:** Python never re-implements recognition. Embedding and matching run through
`cardvision`, a Swift CLI built from the same `CardVision` package the app uses, so every number
here is the number the phone would get.

```
card art ─┐                                   ┌─► data/cards/printings.f32 + .meta.json ─► app
          ├─► generate_embeddings ─(cardvision embed)─┘
scans ────┘                                                    
test set (photos, synth, scans, negatives) ─► evaluate ─(cardvision match)─► report + results.csv
card art ─► train_embedding (PyTorch) ─► export_coreml ─► CardEmbedder.mlpackage ─► app / --model
```

## Setup

```sh
cd ml
uv sync                  # numpy, pillow, requests (+ pytest)
uv sync --extra train    # + torch, torchvision, coremltools (only for fine-tuning)
uv run pytest -q         # lab unit tests
```

The first `cardvision` call builds the Swift CLI (about 30 s). It needs Xcode, and `DEVELOPER_DIR`
is set automatically.

## Walkthrough: from nothing to a report

Each step says what to look at afterwards.

**1. Card data and art**

```sh
uv run scripts/fetch_cards.py            # roster cards + art -> data/cards/, full catalog -> data/cards/catalog.json
```
Look at `data/cards/cards.json` and `printings.json`, which are generated from `data/cards/roster.json`,
and at the art in `data/cards/art/`. Notice the SAMPLE watermark on every API image. It matters (see
"What the numbers mean"). `fetch_cards.py` also writes `data/cards/catalog.json` (every printing, tracked in git). The
full-catalog index needs `--art all` (~1.4 GB, best done on the Mac mini). Add `--install-art` to copy the roster art into the app for image tracking.

**2. A test set**

```sh
uv run scripts/prepare_dataset.py synth          # 14 synthetic photos per roster printing
uv run scripts/prepare_dataset.py negatives      # 150 photos of cards NOT in the roster
uv run scripts/prepare_dataset.py build-test     # -> datasets/test/manifest.json
```
Open `datasets/raw/synth/<printingId>/`. Each image is named after its condition: clean, angle,
glare, dim, blur, upside_down, or small. That's how per-condition accuracy is computed.

**3. Reference embeddings (the "model" with no training)**

```sh
uv run scripts/generate_embeddings.py --min-similarity 0.8
```
This now embeds the whole catalog (about 4.2k rows, a few minutes). `--scope roster` is the old
roster-only index. It writes `data/cards/printings.f32` (one 768-float row per reference image) and
`printings.meta.json` (the backend, one printing ID per row, and the device's rejection threshold).
The next Xcode build bundles both.

**4. Evaluate**

```sh
uv run scripts/evaluate.py --name my-first-run
```
Read `runs/<timestamp>-my-first-run/report.md`:
- **Detection:** did the rectangle detector find the card at all?
- **OCR accuracy:** did OCR read the right code (no read counts as wrong)?
- **Within-group top-1:** given the right code with ≥2 printings, did the embedder pick the right one? This is the number fine-tuning should move.
- **By method:** accuracy for `ocr-unique`, `ocr+vision`, `vision-only`.
- **Top-1 / top-3 printing:** the exact printing (base vs alt art vs manga).
- **Top-1 variant:** the one that matters for the product, meaning which character spawns.
- **By condition / kind / set:** where it breaks. Precision appears for printing attributes.
- Negatives whose printing is in the index are scored as positives ("catalog" source), so the rejection section appears only for a roster-scope index.
- **Rejecting unknown cards:** for each threshold, how many correct matches survive and how many
  non-roster cards get wrongly accepted. The suggested value is what `--min-similarity` should be.
- **Hardest misses:** the most confident wrong answers, which are your hard negatives.

`predictions.jsonl` has the raw ranked candidates for every image, and `metrics.json` has everything
in machine-readable form. Each run appends one row to `results/results.csv`, which is tracked in git
and is the history you compare against.

**5. Change something and compare**

That's the loop. For example, try it without OCR:
```sh
uv run scripts/evaluate.py --name no-ocr --no-ocr
```
Or edit a detector parameter in `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardDetector.swift`
and re-run `evaluate.py`. The CLI rebuilds automatically, and the app gets the same change.

## Real data (the test set that matters)

Synthetic photos come from the same watermarked art as the references, so they overestimate
accuracy. Two ways to get real data:

- **Your own photos:** `datasets/raw/photos/<printingId>/<condition>/*.jpg`, for example
  `photos/OP05-119_p1/glare/IMG_0412.jpg`. Take them on your desk under different lighting and angles.
  Then run `build-test` again.
- **Device scan logs:** the app logs every scan to Documents/Scans. Copy that folder to the Mac through
  Finder (iPhone > Files > OnePieceAR), then run:
  ```sh
  uv run scripts/prepare_dataset.py import-scans ~/Downloads/Scans
  uv run scripts/prepare_dataset.py build-test
  ```
  A stable 30% of scans become test images. The rest can become extra references:
  `generate_embeddings.py --with-scans corrected`. Clean photos of real cards as references are the
  simplest way to beat the watermark.

## Fine-tuning (only once feature prints plateau)

```sh
uv run scripts/fetch_cards.py --art all                # ~4k printings, ~1.4 GB of art
uv run scripts/train_embedding.py --name v1 --epochs 10  # MobileNetV3 + CosFace, uses the Mac GPU (MPS)
uv run scripts/export_coreml.py --name v1                # -> models/v1/CardEmbedder.mlpackage
uv run scripts/generate_embeddings.py --model models/v1/CardEmbedder.mlpackage --out datasets/references/v1.f32
uv run scripts/evaluate.py --name v1 --index datasets/references/v1.f32 --model models/v1/CardEmbedder.mlpackage
```
The training log prints a synthetic validation top-1 per epoch. Negative test cards are held out of
training automatically. If v1 beats the baseline in `results.csv`, ship it:
`export_coreml.py --name v1 --install` copies it into the app, then rebuild the app's index with
`generate_embeddings.py --model … --min-similarity <v1's suggestion>`. The app refuses an index built
with a different model (the backend ID includes the model version).

## Scripts

| Script | Does |
| --- | --- |
| `fetch_cards.py` | OPTCG API -> roster JSON, art, full catalog |
| `prepare_dataset.py` | `synth`, `negatives`, `import-scans`, `build-test` |
| `generate_embeddings.py` | reference index via `cardvision embed` (full catalog by default, `--scope roster` for roster only) |
| `evaluate.py` | metrics via `cardvision match`, report, results history |
| `train_embedding.py` | fine-tune an embedder on augmented card art |
| `export_coreml.py` | checkpoint -> Core ML `CardEmbedder.mlpackage` |

The code lives in `oplab/`, and the scripts are thin entry points. Everything under `datasets/`,
`models/`, and `runs/` is gitignored. `results/results.csv` is tracked.
