import ARKit
import CoreImage
import OnePieceKit

nonisolated struct RecognitionResult: Sendable {
    let crop: CGImage
    let candidates: [RecognitionCandidate]
    let ocrCardID: String?

    var best: RecognitionCandidate? { candidates.first }
}

/// ARKit hands us a CVPixelBuffer on the main actor; recognition reads it on its own actor while
/// the main actor never touches it again.
nonisolated struct PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
}

/// Camera frame -> rectified card -> ranked printings. Runs off the main actor.
actor RecognitionService {
    private let detector = CardDetector()
    private let engine = EmbeddingEngine()
    private let ocr = CardOCR()
    private var matcher: VariantMatcher?

    var referenceSummary: String? {
        guard let matcher else { return nil }
        let source = switch matcher.source {
        case .bundledIndex: PrintingEmbeddings.fileName
        case .computedFromCardArt: "bundled card art"
        }
        return "\(matcher.referenceCount) printings from \(source)"
    }

    /// Loads `printings.f32` if bundled; otherwise computes reference embeddings from any card art
    /// in the bundle, so recognition works for the roster before the ML pipeline exists.
    /// - Returns: `false` when there is nothing to match against.
    func prepare(catalog: CardCatalog, bundledIndex: URL?, cardArt: [(printingID: String, image: CGImage)]) -> Bool {
        if let bundledIndex, let embeddings = try? PrintingEmbeddings.load(from: bundledIndex, catalog: catalog) {
            matcher = VariantMatcher(catalog: catalog, embeddings: embeddings, source: .bundledIndex)
            return true
        }
        let computed = cardArt.compactMap { art in
            (try? engine.embedding(for: art.image)).map { (printingID: art.printingID, vector: $0) }
        }
        guard !computed.isEmpty, let embeddings = try? PrintingEmbeddings.build(from: computed) else { return false }
        matcher = VariantMatcher(catalog: catalog, embeddings: embeddings, source: .computedFromCardArt)
        return true
    }

    /// - Parameter frame: ARKit's captured image (landscape sensor orientation).
    func recognize(_ frame: PixelBufferBox) throws -> RecognitionResult? {
        guard let matcher else { return nil }
        // Portrait UI: the sensor image must be rotated to match what the user sees.
        let image = CIImage(cvPixelBuffer: frame.buffer).oriented(.right)
        guard let upright = try detector.detectCard(in: image) else { return nil }

        // The card might be upside down relative to the phone; keep whichever orientation matches better.
        var best = (crop: upright, matches: matcher.artMatches(for: try engine.embedding(for: upright)))
        if let flipped = detector.rotated180(upright) {
            let flippedMatches = matcher.artMatches(for: try engine.embedding(for: flipped))
            if (flippedMatches.first?.similarity ?? 0) > (best.matches.first?.similarity ?? 0) {
                best = (flipped, flippedMatches)
            }
        }
        guard !best.matches.isEmpty else { return nil }

        var ocrCardID: String?
        if CandidateRanker.needsOCR(best.matches) {
            ocrCardID = try? ocr.cardID(in: best.crop)
        }
        return RecognitionResult(
            crop: best.crop,
            candidates: matcher.rank(best.matches, ocrCardID: ocrCardID),
            ocrCardID: ocrCardID)
    }
}
