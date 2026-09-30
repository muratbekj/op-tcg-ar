import OnePieceKit

/// Art similarity against per-printing reference embeddings, resolved against the full catalog.
public struct VariantMatcher: Sendable {
    public enum Source: Sendable {
        case bundledIndex
        case computedFromCardArt
    }

    public let catalog: FullCatalog
    public let embeddings: PrintingEmbeddings
    /// Where the reference embeddings came from, for the settings/debug screen.
    public let source: Source

    public init(catalog: FullCatalog, embeddings: PrintingEmbeddings, source: Source) {
        self.catalog = catalog
        self.embeddings = embeddings
        self.source = source
    }

    public var referenceCount: Int { embeddings.printingCount }

    /// Top-k printings over the whole index.
    public func artMatches(for embedding: [Float], k: Int = 5) -> [ArtMatch] {
        embeddings.matches(for: embedding, k: k)
    }

    /// Every printing of a code group, ranked by similarity among the group's own rows.
    public func rankGroup(_ group: [CatalogEntry], embedding: [Float]) -> [RecognitionCandidate] {
        let ids = Set(group.map(\.printingId))
        return RecognitionCandidate.rankGroup(group, matches: embeddings.matches(for: embedding, k: ids.count, restrictedTo: ids))
    }

    public func candidates(for matches: [ArtMatch]) -> [RecognitionCandidate] {
        matches.map { match in
            RecognitionCandidate(
                printingID: match.printingID,
                cardID: catalog.entry(id: match.printingID)?.cardId ?? FullCatalog.cardID(ofPrinting: match.printingID),
                similarity: match.similarity)
        }
    }
}
