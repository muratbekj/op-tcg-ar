"""Generate per-printing reference embeddings -> data/cards/printings.f32.

Contract with the app (docs/cv-pipeline.md):
- The app embeds the full perspective-corrected card (630x880, portrait) with
  VNGenerateImageFeaturePrintRequest revision 2. References must use the same extractor on the
  same kind of input, or the similarities are meaningless. Vision is not available from Python,
  so either shell out to a small Swift CLI or switch the app to a Core ML model (export_coreml.py).
- Flat little-endian float32, row-major. Write each printing's row index to `embeddingRow`
  in printings.json.
"""

raise SystemExit("generate_embeddings.py: not implemented yet")
