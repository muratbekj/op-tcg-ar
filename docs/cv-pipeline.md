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
3. **Embed.** `VNGenerateImageFeaturePrintRequest`, **revision 2**, `.scaleFill`, on the full
   rectified card. The 180°-rotated crop is also embedded, and whichever orientation matches
   better wins (for upside-down cards).
4. **Match.** Cosine similarity against the reference rows (`EmbeddingIndex`, vDSP). A printing can
   own several rows and is reported once, at its best row. It returns the top 5. If the best match is
   below the index's `minimumSimilarity` (default 0.80), the frame is rejected and scanning continues.
   This keeps cards outside the roster from spawning a character.
5. **OCR (narrowing only).** Runs when the top two are within 0.04 of each other or the top
   match is below 0.80. It reads the bottom-right 55%×14% of the card, and `CardNumberParser`
   extracts `OP05-119`-style ids. Printings of that card number move to the front. Nothing is
   dropped.
6. **Spawn immediately.** The best guess spawns right away, and "Not this one?" lists the
   ranked alternatives.
7. **Track.** An `ARReferenceImage` (0.063 m wide) is built from the bundled art, or from the
   scanned crop if there's no art. Poses are low-pass filtered in `CardAnchor` (Settings >
   Anchor smoothing).

Frames are sampled every 0.35 s, and only while scanning.

## Embedding contract (`printings.f32` + `printings.meta.json`)

- Flat little-endian float32, row-major. `meta.rows[i]` is row i's printing ID, and IDs may repeat.
- `meta.backend` must equal `EmbeddingEngine.backendID` on the device: `vision-featureprint-r2`, or
  `coreml:CardEmbedder@<version>` when a `CardEmbedder` model is bundled. On a mismatch the app
  ignores the index and computes references from bundled card art instead.
- Rows don't need to be normalized; they're L2-normalized on load.
- References are always produced by `cardvision embed` (`ml/scripts/generate_embeddings.py`), which
  renders each image to the canonical 630×880 card exactly as the device does.
- Legacy: without the meta file, the app falls back to `printings.json[].embeddingRow` (feature prints only).
- `artCrop` is reserved for experiments. The device embeds the full card.

## Baseline (2026-09-25, Vision feature print, synthetic test set)

196 synthetic photos of the 14 roster printings plus 150 non-roster negatives. See `ml/results/results.csv`.

| Detection | Top-1 printing | Top-1 given detected | Top-1 variant | Suggested threshold |
| --- | --- | --- | --- | --- |
| 85.7% | 77.0% | 89.9% | 100% | 0.80 (2% false accepts) |

Most printing-level misses are reprints with identical art (for example `OP06-118_r1` against `_p2`),
which don't change which character spawns. Synthetic photos share the SAMPLE-watermarked art of the
references, so real-card numbers will differ. Photograph your cards to get the real test set.

## Scan logs (learning loop input)

With Settings > Log scans enabled (the default), every recognized scan writes
`Documents/Scans/<yyyyMMdd-HHmmss_id>/crop.jpg` and `scan.json` (`ScanRecord`). The JSON holds
the ranked candidates with similarities, the OCR read, the spawned printing, the final printing,
and `corrected`. Copy the folder to the Mac through Finder (iPhone > Files > OnePieceAR) or the
Files app.
