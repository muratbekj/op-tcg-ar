# Asset pipeline

Assets are Bandai IP. They stay out of git (see `.gitignore`) and out of any distribution.

## Where files go

Drop files into `apps/ios/OnePieceAR/Resources/` and they're bundled automatically
(synchronized folder). You don't need to touch the Xcode project.

| Folder | File name | Used for |
| --- | --- | --- |
| `Characters/` | `<modelAsset>` from variants.json, e.g. `luffy_gear4.usdz` | Model, skinned, T-pose or idle |
| `Animations/` | `<variantId>_<clip>.usdz`, e.g. `luffy_gear4_attack_heavy.usdz` | Per-variant clip (preferred) |
| `Animations/` | `<clip>.usdz`, e.g. `attack_heavy.usdz` | Shared clip, fallback |
| `Cards/` | `<printingId>.png` (or jpg), e.g. `OP05-119_p1.png` | Image tracking + on-device reference embeddings |
| `VFX/` | `vfx_<kind>.usdz`, e.g. `vfx_haki.usdz` | Overrides the procedural particles for that effect |

Clip names come from `variants.json` (`idle`, `walk`, `hit`, `ko`, `victory`, `attacks[].clip`).
For each clip, `AssetService` looks for the per-variant file first, then the shared file. For
idle only, it then falls back to the first animation embedded in the model file. Missing clips
are skipped, and missing models spawn the placeholder rig. The card detail screen shows which
clips were found.

## Per-variant recipe (about half a day)

1. **Pick the art.** Full-body pose, readable silhouette, iconic form.
2. **Image-to-3D.** Meshy, Tripo, Hunyuan3D, or Lens Studio. Generate 3–4 times and keep the best.
3. **Blender cleanup.** Fix the silhouette, decimate under ~50k triangles, bake one 1024
   texture, and apply a T-pose if needed. Face +Z, Y up.
4. **Mixamo.** Upload, auto-rig (humanoid), and download each clip **with skin** as FBX:
   idle, walk, one attack, hit reaction, KO, victory. Use "In Place" for walk and attacks, because
   the app moves the character itself (lunge, wander).
5. **USDZ.** Convert each FBX (Reality Converter, or Blender's USD export, then `usdzip`). Name
   the files as in the table above.
6. **Verify.** Open each file in Reality Composer Pro and check that the animation plays and the
   scale looks sane. The app rescales to `heightMeters` anyway.
7. Set `heightMeters` in variants.json. For reference, a card is 0.088 m tall, so 0.12 m reads well.

### Why per-variant clips

RealityKit binds animations by joint path. A clip exported with skin from the same Mixamo
rig always matches. A shared clip only works if every model's skeleton hierarchy is identical,
which is true for Mixamo auto-rigs in practice but not guaranteed. Start per-variant, and move
clips to the shared library once you've confirmed they bind.

## VFX

Built-in procedural effects: `haki`, `slash`, `fire`, `lightning`, `smoke`, `impact`
(`VFXLibrary.swift`). A variant's attack lists the effects it composes. Adding a new kind means
adding a case to `VFXKind` in OnePieceKit and to `VFXLibrary`.
