import RealityKit
import simd

/// A stage that sits on a physical card (or a tapped surface) in world space. Card poses are
/// low-pass filtered because glossy and foil cards make ARKit's image pose jitter.
final class CardAnchor {
    /// Local frame: origin at the card center, +Y out of the card, +Z toward the card's bottom edge.
    let stage = Entity()
    private(set) var isPlaced = false
    /// Name of the reference image this anchor follows, if any.
    var imageName: String?

    init(parent: Entity) {
        stage.name = "CardStage"
        stage.isEnabled = false
        stage.addChild(StageLighting.makeShadowCatcher())
        parent.addChild(stage)
    }

    /// - Parameter smoothing: 0 snaps to every update, 0.9 is very smooth but laggy.
    func follow(_ worldTransform: simd_float4x4, smoothing: Float) {
        let target = Transform(matrix: worldTransform)
        guard isPlaced else {
            place(at: worldTransform)
            return
        }
        let t = 1 - min(max(smoothing, 0), 0.95)
        var current = stage.transform
        current.translation = simd_mix(current.translation, target.translation, SIMD3(repeating: t))
        current.rotation = simd_slerp(current.rotation, target.rotation, t)
        current.scale = .one
        stage.transform = current
    }

    func place(at worldTransform: simd_float4x4) {
        var transform = Transform(matrix: worldTransform)
        transform.scale = .one
        stage.transform = transform
        stage.isEnabled = true
        isPlaced = true
    }

    func reset() {
        isPlaced = false
        imageName = nil
        stage.isEnabled = false
    }

    func remove() {
        stage.removeFromParent()
    }
}
