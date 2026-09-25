import Foundation
import OnePieceKit
import RealityKit
import simd

/// Behavior for one spawned character: idle loop, wandering around its card, attacks with a lunge,
/// hit reactions with knockback, victory, and KO. The rig animates the body; this moves `root`.
final class CharacterController {
    enum State {
        case idle, walking, busy, knockedOut
    }

    let variant: CharacterVariant
    let root = Entity()
    private(set) var state: State = .idle
    var wanderEnabled = true
    /// Wander stays inside this radius of the card center, in meters.
    var wanderRadius: Float = 0.025
    /// Set in battle so the character faces and lunges at its opponent instead of wandering.
    weak var opponent: CharacterController?

    private let rig: CharacterRig
    private let vfx: VFXLibrary
    private var wanderTask: Task<Void, Never>?
    private var home = Transform.identity

    init(variant: CharacterVariant, rig: CharacterRig, vfx: VFXLibrary) {
        self.variant = variant
        self.rig = rig
        self.vfx = vfx
        root.name = "Character:\(variant.id)"
        root.addChild(rig.entity)
        rig.loop(.idle)
        startWandering()
    }

    var isBusy: Bool { state == .busy || state == .knockedOut }

    // MARK: Actions

    /// Pops the character up off the card.
    func playEntrance() async {
        // Restart idle now that the stage is enabled and in the scene.
        rig.loop(.idle)
        let final = root.transform
        root.scale = SIMD3(repeating: 0.01)
        vfx.play(.smoke, at: .zero, in: root.parent ?? root, scale: 1)
        vfx.play(.impact, at: [0, rig.height * 0.3, 0], in: root.parent ?? root)
        await move(root, to: final, duration: 0.35, timing: .easeOut)
    }

    func attack(_ attack: Attack? = nil) async {
        guard !isBusy, let attack = attack ?? variant.animations.attacks.first else { return }
        state = .busy
        rig.loop(.idle)

        let start = root.transform
        var strike = start
        if let target = opponentPosition() {
            await face(target, duration: 0.15)
            let offset = target - root.position
            let reach = min(simd_length(offset) * 0.45, 0.08)
            strike.translation = root.position + simd_normalize(offset) * reach
            strike.rotation = root.orientation
        } else {
            strike.translation = root.position + root.orientation.act([0, 0, 0.015])
        }
        let effectPoint = strike.translation + root.orientation.act([0, rig.height * 0.5, rig.height * 0.25])

        let clip = Task { await rig.play(.attack(attack)) }
        await move(root, to: strike, duration: 0.18, timing: .easeIn)
        for kind in attack.vfx {
            vfx.play(kind, at: effectPoint, in: root.parent ?? root)
        }
        await clip.value
        await move(root, to: opponent == nil ? start : Transform(scale: .one, rotation: root.orientation, translation: start.translation),
                   duration: 0.3, timing: .easeInOut)
        finishAction()
    }

    /// - Parameter blocked: the hit was countered; flinch without the full reaction.
    func takeHit(blocked: Bool = false) async {
        guard state != .knockedOut else { return }
        state = .busy
        let start = root.transform
        var knockback = start
        let away = opponentPosition().map { simd_normalize(root.position - $0) } ?? root.orientation.act([0, 0, -1])
        knockback.translation += away * (blocked ? 0.004 : 0.012)

        vfx.play(blocked ? .haki : .impact, at: root.position + [0, rig.height * 0.55, 0], in: root.parent ?? root)
        if blocked {
            await move(root, to: knockback, duration: 0.08, timing: .easeOut)
        } else {
            let clip = Task { await rig.play(.hit) }
            await move(root, to: knockback, duration: 0.1, timing: .easeOut)
            await clip.value
        }
        await move(root, to: start, duration: 0.25, timing: .easeInOut)
        finishAction()
    }

    func celebrate() async {
        guard !isBusy else { return }
        state = .busy
        await faceViewer()
        await rig.play(.victory)
        finishAction()
    }

    func knockOut() async {
        state = .busy
        vfx.play(.smoke, at: root.position, in: root.parent ?? root)
        await rig.play(.ko)
        state = .knockedOut
    }

    /// Back to idle at the home spot, e.g. for a rematch.
    func reset() {
        root.stopAllAnimations()
        root.transform = home
        state = .idle
        rig.loop(.idle)
        if let target = opponentPosition() { turnToFace(target) }
    }

    func stop() {
        wanderTask?.cancel()
        wanderTask = nil
    }

    // MARK: Facing

    func faceOpponent() async {
        if let target = opponentPosition() { await face(target, duration: 0.3) }
    }

    /// Mixamo characters face +Z; the card's +Z points at its bottom edge, toward whoever holds it.
    private func faceViewer() async {
        guard opponent == nil else { return }
        var t = root.transform
        t.rotation = .yaw(0)
        await move(root, to: t, duration: 0.25, timing: .easeInOut)
    }

    private func face(_ point: SIMD3<Float>, duration: TimeInterval) async {
        var t = root.transform
        t.rotation = Self.yaw(toward: point - root.position)
        await move(root, to: t, duration: duration, timing: .easeInOut)
    }

    private func turnToFace(_ point: SIMD3<Float>) {
        root.orientation = Self.yaw(toward: point - root.position)
        home = root.transform
    }

    private static func yaw(toward direction: SIMD3<Float>) -> simd_quatf {
        .yaw(atan2(direction.x, direction.z))
    }

    /// The opponent's position in this character's parent (stage) space.
    private func opponentPosition() -> SIMD3<Float>? {
        guard let opponent, let stage = root.parent else { return nil }
        return opponent.root.position(relativeTo: stage)
    }

    private func finishAction() {
        state = .idle
        rig.loop(.idle)
    }

    // MARK: Wandering

    /// Small step-and-turn moves around the card so the character never looks bolted down.
    private func startWandering() {
        wanderTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(Double.random(in: 2.5...5.5)))
                guard let self, !Task.isCancelled else { return }
                guard self.wanderEnabled, self.state == .idle, self.opponent == nil else { continue }
                await self.wanderStep()
            }
        }
    }

    private func wanderStep() async {
        state = .walking
        let angle = Float.random(in: 0..<(2 * .pi))
        let distance = Float.random(in: 0.3...1) * wanderRadius
        let target = SIMD3<Float>(cos(angle) * distance, 0, sin(angle) * distance)
        let travel = simd_length(target - root.position)
        guard travel > 0.004 else {
            state = .idle
            return
        }
        await face(target, duration: 0.25)
        guard state == .walking else { return }
        rig.loop(rig.hasWalkClip ? .walk : .idle)
        var destination = root.transform
        destination.translation = target
        await move(root, to: destination, duration: TimeInterval(travel / 0.035), timing: .easeInOut)
        guard state == .walking else { return }
        rig.loop(.idle)
        await faceViewer()
        if state == .walking { state = .idle }
    }

    // MARK: Helpers

    private func move(_ entity: Entity, to transform: Transform, duration: TimeInterval, timing: AnimationTimingFunction) async {
        entity.move(to: transform, relativeTo: entity.parent, duration: duration, timingFunction: timing)
        try? await Task.sleep(for: .seconds(duration))
    }

    /// Records the current pose as home and faces the opponent. Call after placing both fighters.
    func settle() {
        if let target = opponentPosition() {
            turnToFace(target)
        } else {
            home = root.transform
        }
    }
}
