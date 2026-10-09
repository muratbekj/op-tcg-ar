# One Piece Card Battle AR

An iPhone app that identifies the exact One Piece TCG card in front of the camera, out of 4,212
printings, fully on-device. The goal: the card's character comes alive on top of it in AR.

<table align="center">
  <tr>
    <td align="center"><img src="docs/images/demo.gif" width="250" alt="Apple's Vision feature print identifying base and parallel Sanji"></td>
    <td align="center"><img src="docs/images/demo-v1.gif" width="250" alt="The fine-tuned v1 model identifying base and parallel Sanji, with Settings showing coreml:CardEmbedder@v1"></td>
  </tr>
  <tr>
    <td align="center">Baseline: Apple Vision feature print</td>
    <td align="center">My fine-tuned model (v1), on-device</td>
  </tr>
</table>

| | Apple Vision (baseline) | My model (v1) |
| --- | --- | --- |
| Closest **wrong** card's score, base / parallel | 0.797 / 0.750 | **0.301 / 0.361** |

*Same card number (PRB01-001), different art. Both models pick correctly, but mine leaves a far wider
gap to the runner-up. (These two cards were in v1's training scans.)*

## Why I built this

I'm a software engineering student and a One Piece fan, and I wanted to learn ML by building
something real. This is the first model I've ever fine-tuned, trained on a Mac mini with scans
of my own cards. Telling a base card from its parallel turned out to be the hardest (and most fun) part.

## Results

46 real iPhone scans of cards the model never trained on, matched against all 4,212 printings:

| Model | Top-1 printing (95% CI) | Top-3 |
| --- | --- | --- |
| Apple Vision feature print, no training | 63.0% (48.6–75.5%) | 78.3% |
| **Fine-tuned MobileNetV3 + CosFace (v1)** | **78.3% (64.4–87.7%)** | **87.0%** |

Preliminary: a small set, and the gain is concentrated in one card. Full results and caveats:
[`docs/results.md`](docs/results.md).

## How it works

```
camera frame ─► detect card ─► OCR the card code ─► printings with that code ─► embedding picks one
                                     │ no code read
                                     └──────────────► nearest neighbor over all 4,212 printings
```

- **The hard part:** many printings share a card code and differ only in art (base, parallel, reprints).
- **Fine-tuned embedder:** MobileNetV3 + CosFace in PyTorch on a Mac mini (Apple Silicon), exported to Core ML.
  Printings that share a code go in the same batch, so it learns exactly the differences the app needs.
- **Data loop:** every scan's ✓ or correction becomes a label. Scans are split by printing, so test
  cards never reach training.
- **Ship gate:** a model ships only after it's evaluated on a frozen real-scan test set, and gets a model card.
- **One pipeline:** the app and the offline evals run the same Swift recognition code.

Details: [`docs/cv-pipeline.md`](docs/cv-pipeline.md), [`ml/README.md`](ml/README.md).

## Tech

- **App:** Swift 6, SwiftUI, ARKit, RealityKit, Vision, Core ML (iOS 18+)
- **ML:** Python, PyTorch, coremltools, NumPy, pytest

## Status

Recognition works end-to-end on the phone. AR summoning is in progress. My model runs in a dev build
and ships once the frozen test set (≥200 scans) exists.

## Run it

```sh
make test     # Swift unit tests
make ml-test  # ML lab tests
make open     # open in Xcode, then pick your team under Signing & Capabilities
```

Works without 3D assets (placeholder figures): see [`docs/asset-pipeline.md`](docs/asset-pipeline.md).

---

Unofficial fan project, not affiliated with or endorsed by Bandai, Shueisha, or Toei Animation.
Characters and card art are their IP and are never distributed.
