import Foundation
import OnePieceKit
import RealityKit

/// Plays a variant's clips on a skinned USDZ model. Clips are resolved by `AssetService`, keyed by
/// the clip names in `variants.json`.
final class SkinnedRig: CharacterRig {
    let entity: Entity
    let height: Float
    private let model: Entity
    private let animations: AnimationSet
    private let clips: [String: AnimationResource]
    private var playback: AnimationPlaybackController?

    static let transition: TimeInterval = 0.2

    /// - Parameters:
    ///   - container: normalized wrapper produced by `AssetService` (scaled, grounded, centered).
    ///   - model: the entity that owns the skeleton, where animations must be played.
    init(container: Entity, model: Entity, height: Float, animations: AnimationSet, clips: [String: AnimationResource]) {
        entity = container
        self.model = model
        self.height = height
        self.animations = animations
        self.clips = clips
    }

    var hasWalkClip: Bool { animations.walk.flatMap { clips[$0] } != nil }

    func loop(_ clip: LoopClip) {
        let name = switch clip {
        case .idle: animations.idle
        case .walk: animations.walk ?? animations.idle
        }
        guard let resource = clips[name] ?? clips[animations.idle] else { return }
        playback = model.playAnimation(resource.repeat(), transitionDuration: Self.transition, startsPaused: false)
    }

    func play(_ clip: OneShotClip) async {
        let name: String? = switch clip {
        case .attack(let attack): attack.clip
        case .hit: animations.hit
        case .ko: animations.ko
        case .victory: animations.victory
        }
        guard let name, let resource = clips[name] else { return }
        playback = model.playAnimation(resource, transitionDuration: Self.transition, startsPaused: false)
        // Return slightly early so the caller's next clip blends out of this one's tail.
        try? await Task.sleep(for: .seconds(max(resource.definition.duration - Self.transition, 0.05)))
    }
}
