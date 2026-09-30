import CoreGraphics
import OnePieceKit
import Vision

/// Reads the card number from the bottom-right corner of a canonical card image. The first step
/// of code-first recognition: the code picks the group of printings the embedder chooses among.
public struct CardOCR {
    /// Vision's normalized coordinates, origin bottom-left.
    public static let numberRegion = CGRect(x: 0.45, y: 0.0, width: 0.55, height: 0.14)

    public init() {}

    public func cardID(in card: CGImage) throws -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.regionOfInterest = Self.numberRegion
        try VNImageRequestHandler(cgImage: card, orientation: .up, options: [:]).perform([request])

        for observation in request.results ?? [] {
            for candidate in observation.topCandidates(3) {
                if let id = CardNumberParser.cardID(in: candidate.string) { return id }
            }
        }
        return nil
    }
}
