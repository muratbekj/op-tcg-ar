import CoreImage
import OnePieceKit

public struct RecognitionResult: Sendable {
    public let crop: CGImage
    /// Best first. For `ocrVision`, every printing of the code; for `visionOnly`, the top matches.
    public let candidates: [RecognitionCandidate]
    /// The code OCR read, even when it isn't in the catalog (kept for misread analysis).
    public let ocrCardID: String?
    /// The card was read rotated 180°.
    public let flipped: Bool
    public let method: RecognitionMethod
    /// Printings sharing the read code; 0 for `visionOnly`.
    public let groupSize: Int

    public var best: RecognitionCandidate? { candidates.first }
}

/// The full recognition pipeline, shared by the app and the `cardvision` CLI so offline
/// evaluation measures exactly what runs on the phone.
///
/// Code first: read the card number, then choose among the printings that share it. Falls back to
/// searching the whole index when no catalog code can be read.
public struct CardRecognizer {
    public let engine: EmbeddingEngine
    public let matcher: VariantMatcher
    public var ocrEnabled = true
    /// Candidates returned by the vision-only fallback.
    public var candidateCount = 5
    /// Vision-only best matches below this return `nil` (probably not a card). 0 disables.
    public var minimumSimilarity: Float = RecognitionDefaults.minimumSimilarity

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
        let rotated = CardCanvas.rotated180(upright, context: context)
        let read = ocrEnabled ? readCode(upright: upright, rotated: rotated) : nil

        if let read {
            let group = matcher.catalog.printings(forCode: read.code)
            if group.count == 1 {
                return RecognitionResult(
                    crop: read.crop, candidates: RecognitionCandidate.rankGroup(group, matches: []),
                    ocrCardID: read.code, flipped: read.flipped, method: .ocrUnique, groupSize: 1)
            }
            if group.count > 1 {
                return RecognitionResult(
                    crop: read.crop, candidates: matcher.rankGroup(group, embedding: try engine.embedding(for: read.crop)),
                    ocrCardID: read.code, flipped: read.flipped, method: .ocrVision, groupSize: group.count)
            }
        }

        // Vision only: the card might be upside down, so keep whichever orientation matches better.
        var best = (crop: upright, matches: matcher.artMatches(for: try engine.embedding(for: upright), k: candidateCount), flipped: false)
        if let rotated {
            let rotatedMatches = matcher.artMatches(for: try engine.embedding(for: rotated), k: candidateCount)
            if (rotatedMatches.first?.similarity ?? -1) > (best.matches.first?.similarity ?? -1) {
                best = (rotated, rotatedMatches, true)
            }
        }
        guard let top = best.matches.first, top.similarity >= minimumSimilarity else { return nil }
        return RecognitionResult(
            crop: best.crop, candidates: matcher.candidates(for: best.matches),
            ocrCardID: read?.code, flipped: best.flipped, method: .visionOnly, groupSize: 0)
    }

    /// The card code from the upright crop, else from the 180°-rotated one.
    private func readCode(upright: CGImage, rotated: CGImage?) -> (crop: CGImage, code: String, flipped: Bool)? {
        if let code = try? ocr.cardID(in: upright) { return (upright, code, false) }
        if let rotated, let code = try? ocr.cardID(in: rotated) { return (rotated, code, true) }
        return nil
    }

    /// Embedding of an image of a whole card, as stored in reference indexes.
    public static func referenceEmbedding(for cardImage: CIImage, engine: EmbeddingEngine, context: CIContext) throws -> [Float] {
        guard let card = CardCanvas.render(cardImage, context: context) else { throw EmbeddingError.noOutput }
        return try engine.embedding(for: card)
    }
}
