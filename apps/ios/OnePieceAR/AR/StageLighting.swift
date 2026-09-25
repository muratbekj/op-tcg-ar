import RealityKit
import UIKit

/// Lighting that keeps characters from looking pasted on: ARKit environment texturing supplies
/// image-based light (configured on the session), a key light casts shadows, and an invisible
/// catcher plane under each card receives them.
enum StageLighting {
    static func makeKeyLight() -> Entity {
        let light = DirectionalLight()
        light.name = "KeyLight"
        light.light.intensity = 1800
        light.light.color = UIColor(white: 1, alpha: 1)
        light.shadow = DirectionalLightComponent.Shadow(maximumDistance: 1.5, depthBias: 2)
        // Mostly overhead, slightly from the viewer's side, so shadows fall behind the character.
        light.look(at: .zero, from: [0.3, 1, 0.5], relativeTo: nil)
        return light
    }

    /// Transparent plane that only shows shadows cast onto it.
    static func makeShadowCatcher(size: Float = 0.3) -> Entity {
        let plane = ModelEntity(
            mesh: .generatePlane(width: size, depth: size),
            materials: [OcclusionMaterial(receivesDynamicLighting: true)])
        plane.name = "ShadowCatcher"
        plane.position.y = 0.0005
        return plane
    }
}
