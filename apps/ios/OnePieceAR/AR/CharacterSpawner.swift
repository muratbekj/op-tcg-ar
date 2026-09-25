import OnePieceKit
import RealityKit

/// Builds a character for a variant and puts it on a card stage.
final class CharacterSpawner {
    private let assets: AssetService
    private let vfx: VFXLibrary

    init(assets: AssetService, vfx: VFXLibrary) {
        self.assets = assets
        self.vfx = vfx
    }

    func spawn(_ variant: CharacterVariant, on anchor: CardAnchor) async -> CharacterController {
        let rig = await assets.makeRig(for: variant)
        let controller = CharacterController(variant: variant, rig: rig, vfx: vfx)
        anchor.stage.addChild(controller.root)
        // Collision shapes make the character tappable.
        controller.root.generateCollisionShapes(recursive: true)
        return controller
    }

    func despawn(_ controller: CharacterController) {
        controller.stop()
        controller.root.removeFromParent()
    }
}
