import CoreImage
import OnePieceKit

public struct RecognitionResult: Sendable {
    public let crop: CGImage
    public let candidates: [RecognitionCandidate]
    public let ocrCardID: String?
    /// The card matched better rotated 180°.
    public let flipped: Bool

    public var best: RecognitionCandidate? { candidates.first }
}

/// The full recognition pipeline, shared by the app and the `cardvision` CLI so offline
/// evaluation measures exactly what runs on the phone.
public struct CardRecognizer {
    public let engine: EmbeddingEngine
    public let matcher: VariantMatcher
    public var ocrEnabled = true
    public var candidateCount = 5
    /// Best matches below this return `nil` (probably a card outside the roster). 0 disables.
    public var minimumSimilarity: Float = CandidateRanker.defaultMinimumSimilarity

    private let detector: CardDetector
    private let ocr = CardOCR()
    private let context: CIContext

    public init(engine: EmbeddingEngine, matcher: VariantMatcher, context: CIContext = CIContext()) {
        self.engine = engine
        self.matcher = matcher
        self.context = context
        detector = CardDetector(context: context)
    }

    /// Photo or camera frame: find the card first. `nil` when no card-shaped rectangle is found.
    public func recognize(photo: CIImage) throws -> RecognitionResult? {
        guard let card = try detector.detectCard(in: photo) else { return nil }
        return try recognize(canonicalCard: card)
    }

    /// An image that is already just the card (reference art, logged scan crop).
    public func recognize(cardImage: CIImage) throws -> RecognitionResult? {
        guard let card = CardCanvas.render(cardImage, context: context) else { return nil }
        return try recognize(canonicalCard: card)
    }

    private func recognize(canonicalCard upright: CGImage) throws -> RecognitionResult? {
        // The card might be upside down relative to the camera; keep whichever orientation matches better.
        var best = (crop: upright, matches: matcher.artMatches(for: try engine.embedding(for: upright), k: candidateCount), flipped: false)
        if let rotated = CardCanvas.rotated180(upright, context: context) {
            let rotatedMatches = matcher.artMatches(for: try engine.embedding(for: rotated), k: candidateCount)
            if (rotatedMatches.first?.similarity ?? -1) > (best.matches.first?.similarity ?? -1) {
                best = (rotated, rotatedMatches, true)
            }
        }
        guard let top = best.matches.first, top.similarity >= minimumSimilarity else { return nil }

        var ocrCardID: String?
        if ocrEnabled, CandidateRanker.needsOCR(best.matches) {
            ocrCardID = try? ocr.cardID(in: best.crop)
        }
        return RecognitionResult(
            crop: best.crop,
            candidates: matcher.rank(best.matches, ocrCardID: ocrCardID),
            ocrCardID: ocrCardID,
            flipped: best.flipped)
    }

    /// Embedding of an image of a whole card, as stored in reference indexes.
    public static func referenceEmbedding(for cardImage: CIImage, engine: EmbeddingEngine, context: CIContext) throws -> [Float] {
        guard let card = CardCanvas.render(cardImage, context: context) else { throw EmbeddingError.noOutput }
        return try engine.embedding(for: card)
    }
}
