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
test set (photos, synth, negatives) ─► evaluate ─(cardvision match)─► report + results.csv
card art ─► train_embedding (PyTorch) ─► export_coreml ─► CardEmbedder.mlpackage ─► app / --model
```

## Setup

```sh
cd ml
uv sync                  # numpy, pillow, requests (+ pytest)
uv sync --extra train    # + torch, torchvision, coremltools (only for fine-tuning)
uv run pytest -q         # lab unit tests
```

The first `cardvision` call builds the Swift CLI (about 30 s). It needs a Swift 6 toolchain on macOS 15+:
Xcode if installed (picked automatically), otherwise the Command Line Tools (`xcode-select --install`).

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
- **OCR accuracy** (detected frames only): did OCR read the right code (no read counts as wrong)?
- **OCR read a catalog code** (detected frames only, the `ocr_used` column in results.csv): how often OCR produced a code in the catalog. Before code-first recognition `ocr_used` meant "OCR ran and returned something", so older rows aren't comparable.
- **Within-group top-1** (detected frames only): given the right code with ≥2 printings, did the embedder pick the right one? This is the number fine-tuning should move.
- **By method:** accuracy for `ocr-unique`, `ocr+vision`, `vision-only`.
- Top-1, OCR accuracy, detection, and within-group show a 95% Wilson confidence interval: with 200 test scans at ~75% accuracy it is about ±6 points, so smaller differences between models are noise.
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
- **Device scan logs:** the app logs every scan to Documents/Scans with the answer you gave on the
  phone: ✓ = `confirmed`, choosing another printing = `corrected`, no answer = `none`. Only labeled
  scans are data. Copy the folder to the Mac (Finder > iPhone > Files > OP Card AR), then:
  ```sh
  uv run scripts/prepare_dataset.py import-scans ~/Downloads/Scans   # new scans + refreshed labels
  make status                                                          # labels, split, freeze progress
  ```
  **Split by printing, not by scan:** a stable hash sends ~30% of printings to *test* forever, so no
  photo of a test card is ever used for training (no leakage). Test scans wait in a pool; once it holds
  ≥200 labeled scans across ≥30 printings, `make freeze-test` writes `ml/testsets/test-vN.json` —
  commit it. Every model is then scored on the same frozen set:
  ```sh
  uv run scripts/evaluate.py --testset latest --name <model>
  ```
  Only printings in the recognition index can be frozen (the rest are reported as not freezable). Once the
  phone's logs are deleted, `ml/datasets/raw/scans` is the only full copy of frozen crops: back it up (a
  frozen set whose crops are gone can't be scored).
  Scans arriving later go to the pool for the next version, never into a frozen set. Train-split scans
  can become extra references: `generate_embeddings.py --with-scans labeled`.

  **v0 baseline:** right after freezing `test-v1`, record the Vision feature print on it, before any
  fine-tuned model: `uv run scripts/evaluate.py --testset test-v1 --name v0`.

## Training your own model (Mac mini)

```sh
make train NAME=v1                 # catalog art + your labeled train-side scans (×4), group-aware batches
make export NAME=v1                # -> ml/models/v1/CardEmbedder.mlpackage (backend coreml:CardEmbedder@v1)
make eval NAME=v0                  # once: the Vision feature print baseline on the frozen test set
make eval NAME=v1                  # builds v1's index, scores the frozen set, writes ml/models/v1/MODEL_CARD.md
make ship NAME=v1                  # only if v1 was evaluated on the current frozen set; card -> docs/models/v1.md
make pull-model                    # install into the app, then build in Xcode
```
Training reads only labeled, train-split, unfrozen scans (`dataset.train_records()`). Test printings
never train. `ARGS='--epochs 3 --views 4'` passes options through; `--scan-weight N` changes the
oversampling, `--no-scans` trains on art alone (a useful ablation). Before the first frozen test set,
`make eval NAME=v1 DIAG=1` gives a quick synthetic/photo read. It's never shippable.
Compare versions in `ml/results/results.csv`, and on the model cards' confidence intervals: a
difference smaller than the intervals is noise.

## Two Macs

The MacBook builds the app; the Mac mini holds the datasets and does training, evals, and the
showcase. Git carries code and small text artifacts; `ml/shipped/` (what the app bundles) travels over
**File Sharing**: the MacBook mounts the mini's repo and copies it. No SSH needed.

**Mac mini, once:** clone the repo to `~/github/op-tcg-ar`, install the Command Line Tools
(`xcode-select --install`, ~3 GB; Xcode isn't needed, but macOS 15+ is), install uv
(`curl -LsSf https://astral.sh/uv/install.sh | sh`) and tmux (`brew install tmux`), then
`make ml-setup-train` and `mkdir ~/oplab-inbox`. In System Settings → General → Sharing turn on
**File Sharing** and add two folders: `~/oplab-inbox` (the iPhone drops scans here) and
`~/github/op-tcg-ar` (the MacBook pulls `ml/shipped/` from here). Run `make mini-doctor` until it says
"all set".

**MacBook, once:** Finder → Go → Connect to Server (⌘K) → `smb://<mini>.local` → log in with the
mini's user → pick `op-tcg-ar`; it mounts at `/Volumes/op-tcg-ar`. Then
`cp ml/remote.env.example ml/remote.env` and keep `MINI_SHIPPED=/Volumes/op-tcg-ar/ml/shipped`.

**Check the index once:** Vision's feature print can differ slightly between macOS versions, and the
phone runs its own OS. After both Macs have built the full index (`generate_embeddings.py`), compare
them on the MacBook (with the mini's repo mounted):
```sh
cd ml && uv run scripts/compare_index.py ../data/cards/printings.f32 /Volumes/op-tcg-ar/data/cards/printings.f32
```
"interchangeable" (cosine ≥ 0.99 for every printing) means the mini can build everything it ships.
Otherwise build the shipped index on the MacBook.

**Each labeling session:**
1. iPhone → Files → Browse → ⋯ → Connect to Server → `smb://<mini>.local` → copy On My iPhone →
   OP Card AR → **Scans** into `oplab-inbox`.
2. Mac mini: `make import-scans` (new scans + refreshed labels; originals move to
   `oplab-inbox/imported/`), then `make status`. `oplab-inbox/imported/` is only a safety archive of
   each copy; delete old stamps whenever.

**Training:** on the mini, `make train NAME=v1` (`caffeinate` keeps it awake; run it inside `tmux` if
you want to close the terminal).

**Shipping to the app:** Mac mini `make ship NAME=vN` (any evaluated version, fine-tuned or the
feature print; `make ship-baseline` still records the current feature-print index as v0 before the
first frozen test set exists) → MacBook (share mounted) `make pull-model` → rebuild in Xcode.
`pull-model` refuses a shipment whose model and index disagree, and removes a stale model for a
feature-print shipment. If `pull-model` removed a model, do Product → Clean Build Folder (⇧⌘K) before
rebuilding.

**Optional, SSH instead of File Sharing:** set `MINI_HOST` (e.g. `you@mini.local`) and `MINI_REPO`
instead of `MINI_SHIPPED`, turn on Remote Login on the mini, and authorize the MacBook's key
(`ssh-copy-id`). `pull-model` then uses rsync, and `make train-remote NAME=v1` starts training on the
mini in tmux from the MacBook. Single Mac? `MINI_HOST=local` (with any `MINI_REPO`) makes `pull-model`
read this Mac's own `ml/shipped/`.

## Scripts

| Script | Does |
| --- | --- |
| `fetch_cards.py` | OPTCG API -> roster JSON, art, full catalog |
| `prepare_dataset.py` | `synth`, `negatives`, `import-scans`, `import-inbox`, `build-test`, `status`, `freeze-test` |
| `generate_embeddings.py` | reference index via `cardvision embed` (full catalog by default, `--scope roster` for roster only) |
| `evaluate.py` | metrics via `cardvision match` on the manifest or a frozen test set (`--testset`), report, results history |
| `train_embedding.py` | fine-tune on art + labeled train scans, group-aware batches |
| `export_coreml.py` | checkpoint -> Core ML `CardEmbedder.mlpackage` |
| `ship.py` | `baseline`: ships the feature-print index as v0 into ml/shipped/ |
| `registry.py` | `eval` and `ship` model versions (model cards) |
| `compare_index.py` | compare two indexes built on different Macs (cosine per printing) |
| `remote.py` | `pull-model`, `doctor`, `train-remote` |

The code lives in `oplab/`, and the scripts are thin entry points. Everything under `datasets/`,
`models/`, and `runs/` is gitignored. `results/results.csv` is tracked.
