import CoreImage
import OnePieceKit
import Vision

/// A detected card: the rectified canonical crop and where it was in the image.
public struct DetectedCard: Sendable {
    public let crop: CGImage
    public let quad: CardQuad
}

/// Finds a card-shaped rectangle in a photo or camera frame and returns it perspective-corrected,
/// upright (portrait), at the canonical card size.
public struct CardDetector {
    public static let aspectRange: ClosedRange<Float> = 0.6...0.98

    private let context: CIContext

    public init(context: CIContext = CIContext()) {
        self.context = context
    }

    /// - Parameter image: image already rotated to match what the user sees (portrait).
    public func detect(in image: CIImage) throws -> DetectedCard? {
        let request = VNDetectRectanglesRequest()
        // Wider than the card's 0.716 on the high side: a card tilted away from the camera
        // foreshortens and looks squarer (eval: 29% detection on angled shots with ±0.08).
        request.minimumAspectRatio = Self.aspectRange.lowerBound
        request.maximumAspectRatio = Self.aspectRange.upperBound
        request.minimumSize = 0.2
        request.minimumConfidence = 0.5
        request.quadratureTolerance = 30
        request.maximumObservations = 1

        try VNImageRequestHandler(ciImage: image, options: [:]).perform([request])
        guard let rectangle = request.results?.first, let crop = rectify(image, rectangle) else { return nil }
        let quad = CardQuad(topLeft: rectangle.topLeft, topRight: rectangle.topRight,
                            bottomRight: rectangle.bottomRight, bottomLeft: rectangle.bottomLeft)
        return DetectedCard(crop: crop, quad: quad)
    }

    /// The rectified card only.
    public func detectCard(in image: CIImage) throws -> CGImage? {
        try detect(in: image)?.crop
    }

    private func rectify(_ image: CIImage, _ rectangle: VNRectangleObservation) -> CGImage? {
        let extent = image.extent
        func point(_ normalized: CGPoint) -> CIVector {
            CIVector(x: extent.minX + normalized.x * extent.width, y: extent.minY + normalized.y * extent.height)
        }
        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else { return nil }
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(point(rectangle.topLeft), forKey: "inputTopLeft")
        filter.setValue(point(rectangle.topRight), forKey: "inputTopRight")
        filter.setValue(point(rectangle.bottomLeft), forKey: "inputBottomLeft")
        filter.setValue(point(rectangle.bottomRight), forKey: "inputBottomRight")
        guard var corrected = filter.outputImage else { return nil }

        // A card held sideways comes out landscape; rotate it to portrait. Upside-down is handled
        // by matching both orientations.
        if corrected.extent.width > corrected.extent.height {
            corrected = corrected.oriented(.left)
        }
        return CardCanvas.render(corrected, context: context)
    }
}

/// The canonical card image every embedding is computed from, on device and in the ML lab alike.
public enum CardCanvas {
    public static let size = CGSize(width: 630, height: 880)

    /// Stretches an image of a whole card to the canonical size.
    public static func render(_ image: CIImage, context: CIContext) -> CGImage? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let scaled = image
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: size.width / extent.width, y: size.height / extent.height))
        return context.createCGImage(scaled, from: CGRect(origin: .zero, size: size))
    }

    public static func rotated180(_ image: CGImage, context: CIContext) -> CGImage? {
        let rotated = CIImage(cgImage: image).oriented(.down)
        return context.createCGImage(rotated, from: rotated.extent)
    }
}
