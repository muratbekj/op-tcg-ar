import CoreImage
import OnePieceKit
import Vision

/// Finds a card-shaped rectangle in a camera frame and returns it perspective-corrected, upright
/// (portrait), at a canonical size.
nonisolated struct CardDetector {
    static let outputSize = CGSize(width: 630, height: 880)

    private let context = CIContext(options: [.useSoftwareRenderer: false])

    /// - Parameter image: camera image already rotated to match the screen (portrait).
    func detectCard(in image: CIImage) throws -> CGImage? {
        let request = VNDetectRectanglesRequest()
        let aspect = Float(CardGeometry.aspectRatio)
        request.minimumAspectRatio = aspect - 0.08
        request.maximumAspectRatio = aspect + 0.08
        request.minimumSize = 0.2
        request.minimumConfidence = 0.7
        request.quadratureTolerance = 25
        request.maximumObservations = 1

        try VNImageRequestHandler(ciImage: image, options: [:]).perform([request])
        guard let rectangle = request.results?.first else { return nil }
        return rectify(image, rectangle)
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
        // later by matching both orientations.
        if corrected.extent.width > corrected.extent.height {
            corrected = corrected.oriented(.left)
        }
        let scaled = corrected
            .transformed(by: CGAffineTransform(translationX: -corrected.extent.minX, y: -corrected.extent.minY))
            .transformed(by: CGAffineTransform(
                scaleX: Self.outputSize.width / corrected.extent.width,
                y: Self.outputSize.height / corrected.extent.height))
        return context.createCGImage(scaled, from: CGRect(origin: .zero, size: Self.outputSize))
    }

    func rotated180(_ image: CGImage) -> CGImage? {
        let rotated = CIImage(cgImage: image).oriented(.down)
        return context.createCGImage(rotated, from: rotated.extent)
    }
}
