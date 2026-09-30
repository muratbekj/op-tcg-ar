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
    /// See `RecognitionDefaults.codeArtMargin`.
    public var codeArtMargin: Float = RecognitionDefaults.codeArtMargin

    private let detector: CardDetector
    private let ocr: CardOCR
    private let context: CIContext

    public init(engine: EmbeddingEngine, matcher: VariantMatcher, context: CIContext = CIContext()) {
        self.engine = engine
        self.matcher = matcher
        self.context = context
        ocr = CardOCR(context: context)
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

        if let read, read.inCatalog {
            let group = matcher.catalog.printings(forCode: read.code)
            let embedding = try engine.embedding(for: read.crop)
            let ranked = matcher.rankGroup(group, embedding: embedding)
            if codeMatchesArt(ranked, embedding: embedding) {
                return RecognitionResult(
                    crop: read.crop, candidates: ranked, ocrCardID: read.code, flipped: read.flipped,
                    method: group.count == 1 ? .ocrUnique : .ocrVision, groupSize: group.count)
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

    /// Trust the read code unless the art clearly belongs to a printing outside its group. A group
    /// with no indexed member can't be checked, so its code is trusted.
    private func codeMatchesArt(_ ranked: [RecognitionCandidate], embedding: [Float]) -> Bool {
        guard let groupBest = ranked.compactMap(\.similarity).max(),
              let overall = matcher.artMatches(for: embedding, k: 1).first else { return true }
        return overall.similarity < max(minimumSimilarity, groupBest + codeArtMargin)
    }

    /// The card code from the upright crop, else from the 180°-rotated one. Only a code in the
    /// catalog counts as valid; `inCatalog == false` means the raw read is kept for analysis only.
    private func readCode(upright: CGImage, rotated: CGImage?) -> (crop: CGImage, code: String, flipped: Bool, inCatalog: Bool)? {
        let isValid: (String) -> Bool = { !matcher.catalog.printings(forCode: $0).isEmpty }
        let uprightReads = (try? ocr.cardIDs(in: upright)) ?? []
        var rotatedReads: [String] = []
        if let rotated, !uprightReads.contains(where: isValid) {
            rotatedReads = (try? ocr.cardIDs(in: rotated)) ?? []
        }
        guard let pick = Self.chooseRead(upright: uprightReads, rotated: rotatedReads, isValid: isValid) else { return nil }
        return (pick.flipped ? (rotated ?? upright) : upright, pick.code, pick.flipped, pick.valid)
    }

    /// Picks the read to use: the first valid upright candidate, else the first valid rotated one,
    /// else the first raw read (upright first, else rotated) flagged invalid.
    static func chooseRead(upright: [String], rotated: [String], isValid: (String) -> Bool)
        -> (code: String, flipped: Bool, valid: Bool)? {
        if let code = upright.first(where: isValid) { return (code, false, true) }
        if let code = rotated.first(where: isValid) { return (code, true, true) }
        if let code = upright.first { return (code, false, false) }
        if let code = rotated.first { return (code, true, false) }
        return nil
    }

    /// Embedding of an image of a whole card, as stored in reference indexes.
    public static func referenceEmbedding(for cardImage: CIImage, engine: EmbeddingEngine, context: CIContext) throws -> [Float] {
        guard let card = CardCanvas.render(cardImage, context: context) else { throw EmbeddingError.noOutput }
        return try engine.embedding(for: card)
    }
}
