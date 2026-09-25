import OnePieceKit
import RealityKit
import UIKit

/// Composable one-shot effects. A bundled `vfx_<kind>.usdz` overrides the procedural particles,
/// so authored effects can be dropped in later without code changes.
final class VFXLibrary {
    private var authored: [VFXKind: Entity] = [:]
    private var checkedAuthored = false

    func play(_ kind: VFXKind, at position: SIMD3<Float>, in parent: Entity, scale: Float = 1) {
        loadAuthoredIfNeeded()
        let effect: Entity
        let lifetime: TimeInterval
        if let template = authored[kind] {
            effect = template.clone(recursive: true)
            lifetime = 2
            for animation in effect.availableAnimations { effect.playAnimation(animation) }
        } else {
            var emitter = Self.emitter(for: kind)
            emitter.burst()
            effect = Entity()
            effect.components.set(emitter)
            lifetime = 1.5
        }
        effect.position = position
        effect.scale = SIMD3(repeating: scale)
        parent.addChild(effect)
        Task {
            try? await Task.sleep(for: .seconds(lifetime))
            effect.removeFromParent()
        }
    }

    private func loadAuthoredIfNeeded() {
        guard !checkedAuthored else { return }
        checkedAuthored = true
        for kind in VFXKind.allCases {
            if let url = Bundle.main.url(forResource: "vfx_\(kind.rawValue)", withExtension: "usdz"),
               let entity = try? Entity.load(contentsOf: url) {
                authored[kind] = entity
            }
        }
    }

    private static func emitter(for kind: VFXKind) -> ParticleEmitterComponent {
        var p = ParticleEmitterComponent()
        p.emitterShape = .sphere
        p.emitterShapeSize = [0.01, 0.01, 0.01]
        p.birthLocation = .surface
        p.isEmitting = false
        p.mainEmitter.isLightingEnabled = false
        p.mainEmitter.blendMode = .additive
        p.mainEmitter.opacityCurve = .quickFadeInOut
        p.mainEmitter.dampingFactor = 3
        p.mainEmitter.spreadingAngle = .pi

        func colors(_ start: UIColor, _ end: UIColor) -> ParticleEmitterComponent.ParticleEmitter.ParticleColor {
            .evolving(start: .single(start), end: .single(end))
        }

        switch kind {
        case .impact:
            p.burstCount = 120
            p.speed = 0.35
            p.mainEmitter.size = 0.004
            p.mainEmitter.lifeSpan = 0.35
            p.mainEmitter.color = colors(.white, .orange)
        case .haki:
            p.emitterShape = .torus
            p.emitterShapeSize = [0.03, 0.005, 0.03]
            p.burstCount = 220
            p.speed = 0.2
            p.mainEmitter.size = 0.006
            p.mainEmitter.lifeSpan = 0.6
            p.mainEmitter.blendMode = .alpha
            p.mainEmitter.color = colors(UIColor(red: 0.45, green: 0.1, blue: 0.8, alpha: 1), .black)
        case .slash:
            p.emitterShape = .box
            p.emitterShapeSize = [0.07, 0.002, 0.004]
            p.burstCount = 160
            p.speed = 0.08
            p.mainEmitter.size = 0.003
            p.mainEmitter.lifeSpan = 0.25
            p.mainEmitter.color = colors(.white, .cyan)
        case .lightning:
            p.burstCount = 90
            p.speed = 0.7
            p.speedVariation = 0.4
            p.mainEmitter.size = 0.0025
            p.mainEmitter.lifeSpan = 0.2
            p.mainEmitter.color = colors(.white, UIColor(red: 1, green: 0.95, blue: 0.3, alpha: 1))
        case .fire:
            p.burstCount = 200
            p.speed = 0.08
            p.mainEmitter.size = 0.008
            p.mainEmitter.lifeSpan = 0.7
            p.mainEmitter.acceleration = [0, 0.25, 0]
            p.mainEmitter.sizeMultiplierAtEndOfLifespan = 0.2
            p.mainEmitter.color = colors(.yellow, .red)
        case .smoke:
            p.burstCount = 60
            p.speed = 0.04
            p.mainEmitter.size = 0.02
            p.mainEmitter.lifeSpan = 1.2
            p.mainEmitter.acceleration = [0, 0.03, 0]
            p.mainEmitter.sizeMultiplierAtEndOfLifespan = 2.5
            p.mainEmitter.blendMode = .alpha
            p.mainEmitter.color = colors(UIColor(white: 0.85, alpha: 0.6), UIColor(white: 0.5, alpha: 0))
        }
        return p
    }
}
