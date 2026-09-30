import ARKit
import OnePieceKit
import RealityKit

/// Owns the ARView and ARSession: world tracking, dynamic card reference images, plane raycasts,
/// and forwarding camera frames to recognition while scanning.
@Observable
final class ARSessionManager: NSObject {
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    @ObservationIgnored let arView: ARView
    /// Everything we place hangs off this identity-at-origin anchor, positioned in world space.
    @ObservationIgnored let worldRoot = AnchorEntity(world: .zero)

    @ObservationIgnored var onImageAnchor: ((ARImageAnchor) -> Void)?
    /// Called for every camera frame while set. Keep the handler cheap and never retain the frame.
    @ObservationIgnored var onFrame: ((ARFrame) -> Void)?

    private(set) var trackingMessage: String?
    private(set) var isRunning = false

    @ObservationIgnored private var referenceImages: [String: ARReferenceImage] = [:]

    override init() {
        arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        super.init()
        arView.session.delegate = self
        arView.renderOptions.insert(.disableMotionBlur)
        arView.scene.addAnchor(worldRoot)
        worldRoot.addChild(StageLighting.makeKeyLight())
    }

    func start() {
        guard Self.isSupported, !isRunning else { return }
        run(options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
    }

    func pause() {
        arView.session.pause()
        isRunning = false
    }

    // MARK: Card tracking

    /// Adds a physical card as a detection image. Throws if ARKit judges the image untrackable
    /// (too few features), in which case callers fall back to plane placement.
    func track(_ image: CGImage, name: String) async throws {
        let reference = ARReferenceImage(image, orientation: .up, physicalWidth: CardGeometry.widthMeters)
        reference.name = name
        try await reference.validate()
        referenceImages[name] = reference
        run(options: [])
    }

    func stopTracking(name: String) {
        guard referenceImages.removeValue(forKey: name) != nil else { return }
        for anchor in arView.session.currentFrame?.anchors ?? [] {
            if let image = anchor as? ARImageAnchor, image.referenceImage.name == name {
                arView.session.remove(anchor: anchor)
            }
        }
        run(options: [])
    }

    private func run(options: ARSession.RunOptions) {
        guard Self.isSupported else { return }
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        config.environmentTexturing = .automatic
        config.detectionImages = Set(referenceImages.values)
        config.maximumNumberOfTrackedImages = min(max(referenceImages.count, 1), 4)
        arView.session.run(config, options: options)
    }

    // MARK: Hit testing

    /// Maps normalized camera-sensor coordinates (landscape, origin top-left) to normalized view
    /// coordinates for the portrait UI. `nil` before the first frame.
    func displayTransform(viewportSize: CGSize) -> CGAffineTransform? {
        arView.session.currentFrame?.displayTransform(for: .portrait, viewportSize: viewportSize)
    }

    /// World transform of the horizontal surface under a screen point.
    func raycastSurface(at point: CGPoint) -> simd_float4x4? {
        arView.raycast(from: point, allowing: .estimatedPlane, alignment: .horizontal).first?.worldTransform
    }

    func entity(at point: CGPoint) -> Entity? {
        arView.entity(at: point)
    }
}

extension ARSessionManager: @preconcurrency ARSessionDelegate {
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        forwardImageAnchors(anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        forwardImageAnchors(anchors)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        onFrame?(frame)
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        trackingMessage = switch camera.trackingState {
        case .normal: nil
        case .notAvailable: "Tracking unavailable"
        case .limited(.initializing): "Move the phone slowly to start tracking"
        case .limited(.excessiveMotion): "Slow down"
        case .limited(.insufficientFeatures): "Point at a textured surface with more light"
        case .limited(.relocalizing): "Relocalizing…"
        case .limited: "Tracking limited"
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        trackingMessage = "AR session failed: \(error.localizedDescription)"
        isRunning = false
    }

    private func forwardImageAnchors(_ anchors: [ARAnchor]) {
        for case let image as ARImageAnchor in anchors where image.isTracked {
            onImageAnchor?(image)
        }
    }
}
