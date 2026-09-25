import CoreGraphics
import OnePieceKit
import Vision

/// Reads the card number from the bottom-right corner of a rectified card. Used only to narrow
/// candidates when art matches are close; it never decides a printing on its own.
nonisolated struct CardOCR {
    /// Vision's normalized coordinates, origin bottom-left.
    static let numberRegion = CGRect(x: 0.45, y: 0.0, width: 0.55, height: 0.14)

    func cardID(in card: CGImage) throws -> String? {
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
