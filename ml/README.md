# ML lab (offline, on the Mac)

Nothing on the phone depends on this at runtime. The lab produces files the app bundles, and
it evaluates recognition using scans logged on the device.

```
uv sync                   # core deps
uv sync --extra vision    # PyTorch, OpenCV, FAISS for experiments
```

| Script | Input | Output |
| --- | --- | --- |
| `fetch_cards.py` | OPTCG API | `data/cards/*.json`, `data/cards/art/` |
| `prepare_dataset.py` | card art, device scan logs | `datasets/processed/`, `datasets/test/` |
| `generate_embeddings.py` | card art | `data/cards/printings.f32` + `embeddingRow` in printings.json |
| `evaluate.py` | test set + index | top-1/top-3, per-set and per-rarity precision/recall |
| `export_coreml.py` | fine-tuned embedding model | `.mlpackage` for the app |

All of these are stubs for now. Before writing them, read `docs/cv-pipeline.md` for the
contracts the app expects (especially the embedding contract).

`datasets/` layout (gitignored): `raw/`, `processed/`, `references/`, `test/`.
