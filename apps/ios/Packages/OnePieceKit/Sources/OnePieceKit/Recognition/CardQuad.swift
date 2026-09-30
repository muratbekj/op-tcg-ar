import CoreGraphics

/// Where a card was detected: its corners in Vision's normalized coordinates (origin bottom-left)
/// of the portrait image recognition ran on.
public struct CardQuad: Equatable, Sendable {
    public let topLeft: CGPoint
    public let topRight: CGPoint
    public let bottomRight: CGPoint
    public let bottomLeft: CGPoint

    public init(topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint) {
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
    }

    /// Top-left, top-right, bottom-right, bottom-left.
    public var corners: [CGPoint] { [topLeft, topRight, bottomRight, bottomLeft] }

    /// The corners in the camera sensor's normalized image coordinates (landscape, origin top-left),
    /// which `ARFrame.displayTransform(for:viewportSize:)` maps onto the screen. The portrait image
    /// is the sensor image rotated 90° clockwise (`CIImage.oriented(.right)`), so a portrait point
    /// (x, y) with origin bottom-left came from sensor point (1 - y, 1 - x).
    public var sensorCorners: [CGPoint] {
        corners.map { CGPoint(x: 1 - $0.y, y: 1 - $0.x) }
    }
}
