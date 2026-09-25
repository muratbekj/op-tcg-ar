import Foundation
import ImageIO
import OnePieceKit
import RealityKit

/// Resolves a variant's model, clips, and card art from the app bundle. Assets are gitignored and
/// may be missing; a variant without a model gets a `ProceduralRig` so the app still runs.
///
/// Clip lookup for clip name `attack_heavy` on variant `luffy_gear4`:
///   1. `luffy_gear4_attack_heavy.usdz` (per-variant Mixamo export, the reliable path)
///   2. `attack_heavy.usdz` (shared library; needs the same skeleton hierarchy)
///   3. idle only: the first animation embedded in the model file itself
final class AssetService {
    private struct Template {
        let model: Entity
        let scale: Float
        let offset: SIMD3<Float>
        let clips: [String: AnimationResource]
    }

    private var templates: [String: Template] = [:]
    private var artCache: [String: CGImage] = [:]

    // MARK: Lookup

    func modelURL(for variant: CharacterVariant) -> URL? {
        Self.resource(variant.modelAsset)
    }

    func hasModel(for variant: CharacterVariant) -> Bool {
        modelURL(for: variant) != nil
    }

    func clipURL(_ clip: String, for variant: CharacterVariant) -> URL? {
        Self.resource("\(variant.id)_\(clip).usdz") ?? Self.resource("\(clip).usdz")
    }

    /// Card art named after the printing ID, e.g. `OP05-119_p1.png`, used for image tracking and
    /// for computing reference embeddings when no `printings.f32` is bundled.
    func cardArt(for printing: Printing) -> CGImage? {
        if let cached = artCache[printing.id] { return cached }
        for ext in ["png", "jpg", "jpeg", "webp"] {
            guard let url = Self.resource("\(printing.id).\(ext)"),
                  let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { continue }
            artCache[printing.id] = image
            return image
        }
        return nil
    }

    // MARK: Rigs

    func makeRig(for variant: CharacterVariant) async -> CharacterRig {
        guard let template = await template(for: variant) else {
            return ProceduralRig(variant: variant)
        }
        let model = template.model.clone(recursive: true)
        model.scale = SIMD3(repeating: template.scale)
        model.position = template.offset
        let container = Entity()
        container.name = "Model:\(variant.id)"
        container.addChild(model)
        return SkinnedRig(
            container: container, model: model, height: Float(variant.heightMeters),
            animations: variant.animations, clips: template.clips)
    }

    private func template(for variant: CharacterVariant) async -> Template? {
        if let cached = templates[variant.id] { return cached }
        guard let url = modelURL(for: variant) else { return nil }
        do {
            let model = try await Entity(contentsOf: url)
            // Normalize: scale to the variant's height, stand on y = 0, center over the origin.
            let bounds = model.visualBounds(relativeTo: nil)
            let scale = bounds.extents.y > 0 ? Float(variant.heightMeters) / bounds.extents.y : 1
            let offset = SIMD3<Float>(-bounds.center.x, -bounds.min.y, -bounds.center.z) * scale

            var clips: [String: AnimationResource] = [:]
            for clip in variant.animations.allClips {
                if let clipURL = clipURL(clip, for: variant),
                   let animation = try? await Entity(contentsOf: clipURL).availableAnimations.first {
                    clips[clip] = animation
                }
            }
            if clips[variant.animations.idle] == nil, let embedded = model.availableAnimations.first {
                clips[variant.animations.idle] = embedded
            }

            let template = Template(model: model, scale: scale, offset: offset, clips: clips)
            templates[variant.id] = template
            return template
        } catch {
            print("AssetService: failed to load \(url.lastPathComponent): \(error)")
            return nil
        }
    }

    /// Bundle resources may be flattened to the bundle root or kept in their folders.
    private static func resource(_ fileName: String) -> URL? {
        let name = (fileName as NSString).deletingPathExtension
        let ext = (fileName as NSString).pathExtension
        for subdirectory in [nil, "Characters", "Animations", "Cards", "VFX"] {
            if let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory) {
                return url
            }
        }
        return nil
    }
}
