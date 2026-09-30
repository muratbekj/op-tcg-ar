# Recognition pipeline

Implemented in the `CardVision` package target (`apps/ios/Packages/OnePieceKit/Sources/CardVision/`),
which is shared by the app (`Services/RecognitionService.swift`) and the ML lab's `cardvision` CLI, so
offline evaluation measures exactly what runs on the phone. The math lives in `OnePieceKit/Recognition/`.
Everything here is unit-tested on the Mac, including real Vision detection on synthetic images.

1. **Detect.** `VNDetectRectanglesRequest` on the portrait-rotated camera frame. The aspect range is
   0.60–0.98, well above the card's 0.716, because a card tilted away from the camera looks squarer.
   Minimum size is 20%, confidence 0.5, and there's one observation. Both values were tuned with
   `evaluate.py`: ±0.08 detected only 29% of angled shots.
2. **Rectify.** `CIPerspectiveCorrection` to 630×880 portrait. Landscape results are rotated.
3. **Read the code.** `CardOCR` crops the bottom-right 55%×14% of the card, enlarges it 3×, and
   `CardNumberParser` extracts `OP05-119`-style codes, tolerating OCR digit confusions (S→5, O→0,
   I→1, B→8, …) and a missing dash. The first candidate that exists in the catalog wins. If the upright crop has none, the
   180°-rotated crop is tried, and the orientation that produced a code is used from then on.
4. **Look up the group.** `FullCatalog` (`catalog.json`, every printing in the OPTCG API) lists the
   printings that share the code. A code that isn't in the catalog counts as no code.
5. **Pick the printing.**
   - One printing: that's the answer, with no ranking (`ocr-unique`).
   - Several: embed the crop (`VNGenerateImageFeaturePrintRequest` revision 2, or the Core ML
     embedder) and rank only the group's index rows. Every member is returned (`ocr+vision`).
   - The art must not contradict the code: the crop is embedded, and if a printing outside the group matches it at least `codeArtMargin` (0.08) better than the group's best (and above `minimumSimilarity`), the code is treated as a misread and the frame goes to the vision-only search, keeping the raw read.
   - No code: embed both orientations, search the whole index, and keep the better orientation's
     top 5 (`vision-only`). Below the index's `minimumSimilarity` (default 0.80), the frame is
     rejected and scanning continues. The threshold never applies when a catalog code was read.
6. **Spawn or show.** A roster printing spawns its character right away; any other printing shows a
   card-info panel (name, code, set, kind, rarity). Either way a strip reads "OP05-119 · 4 printings"
   (or "Matched by art · 5 candidates") with the pick; ✓ confirms it, and tapping the strip opens the
   group list, where any row can be chosen instead ("None of these" opens the manual picker).
   Thumbnails there load on demand from the printing's art URL and are cached; offline, rows are text only.
7. **Track.** An `ARReferenceImage` (0.063 m wide) is built from the bundled art, or from the
   scanned crop if there's no art. Poses are low-pass filtered in `CardAnchor` (Settings >
   Anchor smoothing).

Frames are sampled every 0.35 s, and only while scanning. Settings > Show recognition debug draws the
detected card outline live while scanning (yellow: found, no match; green: recognized) and lists the
OCR read, method, and top similarities.

## Embedding contract (`printings.f32` + `printings.meta.json`)

- Flat little-endian float32, row-major. `meta.rows[i]` is row i's printing ID, and IDs may repeat.
- `meta.backend` must equal `EmbeddingEngine.backendID` on the device: `vision-featureprint-r2`, or
  `coreml:CardEmbedder@<version>` when a `CardEmbedder` model is bundled. On a mismatch the app
  ignores the index and computes references from bundled card art instead.
- Rows don't need to be normalized; they're L2-normalized on load.
- References are always produced by `cardvision embed` (`ml/scripts/generate_embeddings.py`), which
  renders each image to the canonical 630×880 card exactly as the device does.
- Legacy: without the meta file, the app falls back to `printings.json[].embeddingRow` (feature prints only).
- The bundled index covers the full catalog (`generate_embeddings.py` default scope). `catalog.json` is bundled next to it; without it the app recognizes roster printings only.
- `artCrop` is reserved for experiments. The device embeds the full card.

## Baseline (2026-09-25, Vision feature print, synthetic test set)

196 synthetic photos of the 14 roster printings plus 150 non-roster negatives. See `ml/results/results.csv`.

| Detection | Top-1 printing | Top-1 given detected | Top-1 variant | Suggested threshold |
| --- | --- | --- | --- | --- |
| 85.7% | 77.0% | 89.9% | 100% | 0.80 (2% false accepts) |

Most printing-level misses are reprints with identical art (for example `OP06-118_r1` against `_p2`),
which don't change which character spawns. Synthetic photos share the SAMPLE-watermarked art of the
references, so real-card numbers will differ. Photograph your cards to get the real test set.

Comparability: the 2026-09-25 numbers are roster-only (14 printings indexed, 196 synthetic images).
The code-first run indexes the full catalog (~4.2k printings) and adds 150 catalog-card images, so its
top-1 isn't directly comparable to this baseline.

OCR hardening, same 346 images: Phase 1 baseline 42.9% OCR accuracy / 74.6% top-1; the tolerant parser
alone 35.1% / 74.0% (`results.csv` row `code-first-featureprint`); plus crop + 3× upscale 50.3% / 74.0%
(row `ocr-hardened`). Top-1 is flat because wrong-but-real catalog codes cancel the gains (an art-check guard follows).

Art-check guard (`codeArtMargin` 0.08, row `art-guard-0.08`), same 346 images: 76.3% top-1 (from 74.0%), 50.3% OCR
accuracy, 88.6% within-group. Margins 0.05 and 0.08 tie on top-1; 0.12 gives 76.0%. The larger tied margin is kept.

The code-first results are in `ml/results/results.csv`, with the `ocr_accuracy` and `within_group` columns.

## Scan logs (learning loop input)

With Settings > Log scans enabled (the default), every recognized scan writes
`Documents/Scans/<yyyyMMdd-HHmmss_id>/crop.jpg` and `scan.json` (`ScanRecord`). The JSON holds the ranked candidates with similarities, the OCR read, `method` and `groupSize`, the first guess (`spawnedPrintingID`), the final printing, and `label`: `confirmed` (✓ or re-choosing the first guess), `corrected` (another printing chosen), or `none` (no answer). `corrected` is kept as a boolean for older readers. Only labeled scans are meant for training and testing. Copy the folder to the Mac through Finder (iPhone > Files > OnePieceAR) or the
Files app.
