import OnePieceKit
import RealityKit

enum LoopClip {
    case idle, walk
}

enum OneShotClip {
    case attack(Attack), hit, ko, victory
}

/// What drives a character's body. `SkinnedRig` plays Mixamo clips from USDZ; `ProceduralRig` is
/// the primitive placeholder used until a variant's model is bundled.
protocol CharacterRig: AnyObject {
    /// Standing on the origin, facing +Z, already scaled to the variant's height.
    var entity: Entity { get }
    var height: Float { get }
    var hasWalkClip: Bool { get }

    func loop(_ clip: LoopClip)
    /// Plays a one-shot clip and returns when it ends (or immediately if the rig lacks it).
    func play(_ clip: OneShotClip) async
}
