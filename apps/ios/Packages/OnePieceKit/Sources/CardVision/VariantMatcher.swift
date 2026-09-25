import OnePieceKit

/// Art similarity against per-printing reference embeddings, plus OCR narrowing.
public struct VariantMatcher: Sendable {
    public enum Source: Sendable {
        case bundledIndex
        case computedFromCardArt
    }

    public let catalog: CardCatalog
    public let embeddings: PrintingEmbeddings
    /// Where the reference embeddings came from, for the settings/debug screen.
    public let source: Source

    public init(catalog: CardCatalog, embeddings: PrintingEmbeddings, source: Source) {
        self.catalog = catalog
        self.embeddings = embeddings
        self.source = source
    }

    public var referenceCount: Int { embeddings.printingCount }

    public func artMatches(for embedding: [Float], k: Int = 5) -> [ArtMatch] {
        embeddings.matches(for: embedding, k: k)
    }

    public func rank(_ matches: [ArtMatch], ocrCardID: String?) -> [RecognitionCandidate] {
        CandidateRanker.rank(matches, ocrCardID: ocrCardID, catalog: catalog)
    }
}
