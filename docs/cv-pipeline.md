# Recognition pipeline

Implemented in `apps/ios/OnePieceAR/ComputerVision/` and `Services/RecognitionService.swift`.
The math lives in `OnePieceKit/Recognition/` and is unit-tested.

1. **Detect.** `VNDetectRectanglesRequest` on the portrait-rotated camera frame. Aspect is
   0.716 ± 0.08, minimum size is 20% of the frame, and there's one observation.
2. **Rectify.** `CIPerspectiveCorrection` to 630×880 portrait. Landscape results are rotated.
3. **Embed.** `VNGenerateImageFeaturePrintRequest`, **revision 2**, `.scaleFill`, on the full
   rectified card. The 180°-rotated crop is also embedded, and whichever orientation matches
   better wins (for upside-down cards).
4. **Match.** Cosine similarity against per-printing references (`EmbeddingIndex`, vDSP), top 5.
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

## Embedding contract (`printings.f32`)

- Flat little-endian float32, row-major, one row per printing.
- `printings.json[].embeddingRow` is the row index. The row count is `max(embeddingRow) + 1`,
  and the dimension is `floats / rows`. Unlabeled rows are allowed and skipped.
- Rows don't need to be normalized; the app L2-normalizes them on load.
- **References must come from the same extractor as the device:** Vision feature print
  revision 2 on the full rectified card image. Python can't call Vision, so generate them with a
  small Swift CLI (Vision runs on macOS), or replace `EmbeddingEngine` with a Core ML model and
  regenerate all references with that model.
- `artCrop` is recorded per printing for ML experiments. The device embeds the full card. If
  you switch references to art-window crops, crop the query the same way in
  `RecognitionService` first.

## Scan logs (learning loop input)

With Settings > Log scans enabled (the default), every recognized scan writes
`Documents/Scans/<yyyyMMdd-HHmmss_id>/crop.jpg` and `scan.json` (`ScanRecord`). The JSON holds
the ranked candidates with similarities, the OCR read, the spawned printing, the final printing,
and `corrected`. Copy the folder to the Mac through Finder (iPhone > Files > OnePieceAR) or the
Files app.
