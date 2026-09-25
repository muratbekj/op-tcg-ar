import OnePieceKit

/// Art similarity against per-printing reference embeddings, plus OCR narrowing.
nonisolated struct VariantMatcher: Sendable {
    let catalog: CardCatalog
    let embeddings: PrintingEmbeddings
    /// Where the reference embeddings came from, for the settings/debug screen.
    let source: Source

    enum Source: Sendable {
        case bundledIndex
        case computedFromCardArt
    }

    var referenceCount: Int { embeddings.printingIDs.compactMap { $0 }.count }

    func artMatches(for embedding: [Float], k: Int = 5) -> [ArtMatch] {
        embeddings.matches(for: embedding, k: k)
    }

    func rank(_ matches: [ArtMatch], ocrCardID: String?) -> [RecognitionCandidate] {
        CandidateRanker.rank(matches, ocrCardID: ocrCardID, catalog: catalog)
    }
}
