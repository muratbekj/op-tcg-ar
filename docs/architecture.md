# Architecture

Swift is the product. Python is the offline lab on the Mac. Nothing on the phone talks to a server.

```
apps/ios/
├── OnePieceAR.xcodeproj        synchronized folders: new files under OnePieceAR/ are picked up automatically
├── Info.plist                  only keys Xcode can't generate (UIFileSharingEnabled); the rest are build settings
├── Packages/OnePieceKit/       pure Swift, no ARKit/RealityKit/Vision; `make test` runs on the Mac
│   ├── OnePieceKit             Card, Printing, CharacterVariant, CardCatalog, EmbeddingIndex,
│   │                           CardNumberParser, CandidateRanker
│   └── BattleKit               BattleEngine, Fighter, BattleRules
└── OnePieceAR/
    ├── App/                    OnePieceARApp, AppModel (orchestration)
    ├── AR/                     ARSessionManager, CardAnchor, CharacterSpawner, CharacterController,
    │                           CharacterRig (SkinnedRig, ProceduralRig), VFXLibrary, StageLighting
    ├── ComputerVision/         CardDetector, EmbeddingEngine, CardOCR, VariantMatcher
    ├── Services/               AssetService, RecognitionService, ScanLogger
    ├── Features/               Scanner, Collection, CardDetail, ARBattle, Settings
    ├── Models/                 AppSettings, ScanRecord
    └── Resources/              Assets.xcassets (+ gitignored Characters/, Animations/, VFX/, Cards/)
```

`data/cards/*.json` lives at the repo root and is copied into the bundle by the
"Bundle card data" build phase, so the ML scripts and the app share one source of truth.

## Data flow

```
ARKit frame ──► CardDetector ──► EmbeddingEngine ──► VariantMatcher ──► CandidateRanker
 (scanning)     rectangle +       Vision feature     cosine top-k        OCR narrows
                perspective fix   print               (vDSP)             (never decides)
                                                                            │
      ┌─────────────────────────────────────────────────────────────────────┘
      ▼
 printing ──► CardCatalog.variant(for:) ──► CharacterSpawner ──► AssetService ──► rig
                                                   │                               (USDZ or placeholder)
                                                   ▼
                                  CardAnchor (ARImageAnchor or surface tap, smoothed)
```

## Key decisions

- **Everything in the package is testable without a device.** The catalog, the matching math, and
  the battle rules have no AR imports. The app target only holds the parts that need a camera.
- **Missing assets never block.** A variant without a USDZ spawns a `ProceduralRig`, a blocky
  stand-in with the same controller, so AR, anchoring, battle, and recognition can all be tested
  before the asset pipeline produces anything.
- **Recognition works before the ML pipeline exists.** If there's no `printings.f32`,
  `RecognitionService` computes reference embeddings on launch from any card art bundled in
  `Resources/Cards/`. That's fine for a roster of a handful of printings.
- **Anchoring falls back gracefully.** Tracking uses the bundled card art first, then the scanned
  crop, then a tap on a surface. A tap on a surface always works, even while the app is waiting
  for a card image.
- **Controllers move, rigs animate.** `CharacterController` moves the root (wander, lunge,
  knockback, facing) and the rig plays clips on the model. They never write the same transform.
- **Default MainActor isolation** in the app target. CV types that run on `RecognitionService`'s
  actor are marked `nonisolated`.

## Milestone status

| Milestone | State |
| --- | --- |
| M0 repo | Done |
| M1 AR + one character | Code done: anchoring, idle/attack/hit/victory, shadows, IBL, wander. Needs a real USDZ and a device test |
| M2 character system | Done: variants, AssetService, CharacterSpawner, clip resolution |
| M3 card system | Done: catalog, printing override, manual picker |
| M4 recognition | Done on device: detection, feature-print match, OCR narrowing, "not this one?". Needs a device test |
| M5 learning loop | Device side done (ScanLogger + Files export). Python side not started |
| M6 battle | Done: BattleKit rules + tests, two-card HUD with DON!! window |
