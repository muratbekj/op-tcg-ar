# Results

All numbers come from `make eval` (see [`ml/README.md`](../ml/README.md)) and are logged in
[`ml/results/results.csv`](../ml/results/results.csv).

## Demo: base vs parallel Sanji

Same card number (PRB01-001), different art: the scanner tells the base and parallel Sanji apart.
These are Japanese printings, matched against English catalog art, and OCR didn't read the code,
so both picks come from the art alone. Both models choose correctly. The difference is in
the gap: the baseline scores a wrong card almost as high as the right one, while v1 leaves a wide
margin. Each model scores on its own scale, so compare the gaps, not the top scores.
Caveat: these two cards were in v1's training scans, so the clip shows the margin, not how well
it generalizes. For that, see the [real-scan results](#real-phone-scans-preliminary).

## Real phone scans (preliminary)

46 labeled iPhone scans of 8 printings that were never used for training, scored through the full
pipeline against all 4,212 printings (`make eval SCANS=test`):

| Model | Top-1 printing (95% CI) | Top-3 | Top-1 card number |
| --- | --- | --- | --- |
| v0: Vision feature print, no training | 63.0% (48.6–75.5%) | 78.3% | 67.4% |
| **v1: fine-tuned MobileNetV3 + CosFace** | **78.3% (64.4–87.7%)** | **87.0%** | **84.8%** |

On the same scans, v1 fixed 8 that v0 got wrong and broke 1. Read this with care:
- **The gain is concentrated:** 7 of the 8 fixes are scans of one card (OP01-120). Scans of the
  same card aren't independent, so this is a promising signal, not a proven win.
- **One card is still hard for both models** (OP11-067: 1 of 7).
- **OCR read a code on only 20% of these scans,** so most predictions came from the embedding alone.
  Better OCR on real cards is the next lever.

This pool isn't frozen and grows as I label scans. The headline comparison will use the first
frozen test set (≥200 scans across ≥30 printings), which is also what `make ship` requires.

## Synthetic photos and catalog images

Optimistic: synthetic photos are made from the same art as the references.

| Setup | Index | Test images | Top-1 printing | Top-3 | Within-group |
| --- | --- | --- | --- | --- | --- |
| Vision feature print | 14 roster printings | 196 | 77.0% | 84.7% | — |
| Code-first + art guard, feature print | ~4.2k printings | 346 | 76.3% | 87.3% | 88.6% |

The fine-tuned embedder (v1: 4,212 printings + 156 real scans, 3 epochs on a Mac mini M4) reached
**96.3%** top-1 at the end of training. That is embedding-only nearest neighbor on augmented catalog
art (2 views per printing), not the full pipeline above, so the two numbers aren't comparable. The
real-scan comparison is the one that counts.

![v1 training: top-1 accuracy and loss per epoch](images/v1-training.png)

Full history: [`ml/results/results.csv`](../ml/results/results.csv).
