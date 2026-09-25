import OnePieceKit
import RealityKit
import UIKit

/// A blocky stand-in figure with procedural motion, so the AR loop is testable before any
/// USDZ exists. Colors derive from the variant ID so different variants read as different.
final class ProceduralRig: CharacterRig {
    let entity = Entity()
    let height: Float
    let hasWalkClip = true
    /// Animated child; the controller moves `entity`, the rig moves `body`, so they never fight.
    private let body = Entity()

    init(variant: CharacterVariant) {
        height = Float(variant.heightMeters)
        entity.name = "Placeholder:\(variant.id)"
        entity.addChild(body)
        buildFigure(palette: Palette(seed: variant.id))
    }

    func loop(_ clip: LoopClip) {
        body.stopAllAnimations()
        body.transform = .identity
        let (lift, period): (Float, TimeInterval) = switch clip {
        case .idle: (0.03, 0.9)
        case .walk: (0.06, 0.22)
        }
        let bob = FromToByAnimation<Transform>(
            from: .identity,
            to: Transform(scale: [1.02, 0.97, 1.02], translation: [0, height * lift, 0]),
            duration: period,
            timing: .easeInOut,
            bindTarget: .transform,
            repeatMode: .autoReverse)
        if let resource = try? AnimationResource.generate(with: bob) {
            body.playAnimation(resource)
        }
    }

    func play(_ clip: OneShotClip) async {
        body.stopAllAnimations()
        switch clip {
        case .attack:
            await pose(Transform(rotation: .pitch(-0.35), translation: [0, 0, -height * 0.08]), 0.14)
            await pose(Transform(scale: [1.1, 0.95, 1.1], rotation: .pitch(0.55), translation: [0, 0, height * 0.12]), 0.08)
            await pose(.identity, 0.22)
        case .hit:
            await pose(Transform(rotation: .pitch(-0.45), translation: [0, height * 0.03, -height * 0.05]), 0.07)
            await pose(Transform(rotation: .pitch(0.1)), 0.12)
            await pose(.identity, 0.18)
        case .victory:
            for turn in 1...2 {
                await pose(Transform(rotation: .yaw(.pi * Float(turn)), translation: [0, height * 0.35, 0]), 0.22)
                await pose(Transform(rotation: .yaw(.pi * Float(turn))), 0.18)
            }
            body.transform = .identity
        case .ko:
            await pose(Transform(rotation: .pitch(-.pi / 2), translation: [0, height * 0.12, -height * 0.3]), 0.45)
        }
    }

    private func pose(_ transform: Transform, _ duration: TimeInterval) async {
        body.move(to: transform, relativeTo: entity, duration: duration, timingFunction: .easeInOut)
        try? await Task.sleep(for: .seconds(duration))
    }

    // MARK: Figure

    private struct Palette {
        let outfit: UIColor, accent: UIColor, skin: UIColor

        init(seed: String) {
            let hue = CGFloat(abs(seed.hashValueStable) % 360) / 360
            outfit = UIColor(hue: hue, saturation: 0.75, brightness: 0.85, alpha: 1)
            accent = UIColor(hue: (hue + 0.5).truncatingRemainder(dividingBy: 1), saturation: 0.6, brightness: 0.95, alpha: 1)
            skin = UIColor(red: 0.96, green: 0.8, blue: 0.66, alpha: 1)
        }
    }

    private func buildFigure(palette: Palette) {
        let h = height
        func part(_ mesh: MeshResource, _ color: UIColor, _ position: SIMD3<Float>) {
            let material = SimpleMaterial(color: color, roughness: 0.6, isMetallic: false)
            let model = ModelEntity(mesh: mesh, materials: [material])
            model.position = position
            body.addChild(model)
        }
        let legH = h * 0.42, torsoH = h * 0.33, headR = h * 0.12
        let legW = h * 0.1, torsoW = h * 0.32, armW = h * 0.08
        for side in [-1, 1] as [Float] {
            part(.generateBox(width: legW, height: legH, depth: legW, cornerRadius: legW * 0.3), palette.accent,
                 [side * torsoW * 0.25, legH / 2, 0])
            part(.generateBox(width: armW, height: torsoH * 0.95, depth: armW, cornerRadius: armW * 0.4), palette.skin,
                 [side * (torsoW / 2 + armW * 0.6), legH + torsoH * 0.5, 0])
        }
        part(.generateBox(width: torsoW, height: torsoH, depth: torsoW * 0.55, cornerRadius: h * 0.03), palette.outfit,
             [0, legH + torsoH / 2, 0])
        part(.generateSphere(radius: headR), palette.skin, [0, legH + torsoH + headR * 0.95, 0])
        // Visor on the +Z face so facing direction is readable.
        part(.generateBox(width: headR * 1.3, height: headR * 0.35, depth: headR * 0.3, cornerRadius: headR * 0.1),
             palette.accent, [0, legH + torsoH + headR, headR * 0.85])
    }
}

private extension String {
    /// `hashValue` is randomized per launch; this stays stable so colors don't change between runs.
    var hashValueStable: Int {
        unicodeScalars.reduce(5381) { ($0 &* 33) &+ Int($1.value) }
    }
}

extension simd_quatf {
    static func yaw(_ angle: Float) -> simd_quatf { simd_quatf(angle: angle, axis: [0, 1, 0]) }
    static func pitch(_ angle: Float) -> simd_quatf { simd_quatf(angle: angle, axis: [1, 0, 0]) }
}
