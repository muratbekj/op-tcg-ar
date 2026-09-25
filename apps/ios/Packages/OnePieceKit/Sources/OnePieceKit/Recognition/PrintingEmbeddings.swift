import Foundation

/// An embedding index whose rows are labeled with printing IDs.
public struct PrintingEmbeddings: Sendable {
    public static let fileName = "printings.f32"

    public let index: EmbeddingIndex
    /// `printingIDs[row]`; `nil` for rows no printing points at.
    public let printingIDs: [String?]

    public init(index: EmbeddingIndex, printingIDs: [String?]) {
        precondition(printingIDs.count == index.rowCount, "one label per row")
        self.index = index
        self.printingIDs = printingIDs
    }

    /// Loads `printings.f32`, using each printing's `embeddingRow` as the row label.
    /// Row count is `max(embeddingRow) + 1`; the dimension is inferred from the file size.
    public static func load(from url: URL, catalog: CardCatalog) throws -> PrintingEmbeddings {
        let rows = catalog.printings.compactMap { printing in printing.embeddingRow.map { ($0, printing.id) } }
        guard let maxRow = rows.map(\.0).max() else { throw PrintingEmbeddingsError.noEmbeddingRows }
        let index = try EmbeddingIndex(contentsOf: url, rowCount: maxRow + 1)
        var labels = [String?](repeating: nil, count: index.rowCount)
        for (row, id) in rows { labels[row] = id }
        return PrintingEmbeddings(index: index, printingIDs: labels)
    }

    /// Builds an index from embeddings computed at runtime (e.g. from bundled card art).
    public static func build(from embeddings: [(printingID: String, vector: [Float])]) throws -> PrintingEmbeddings {
        let index = try EmbeddingIndex(rows: embeddings.map(\.vector))
        return PrintingEmbeddings(index: index, printingIDs: embeddings.map(\.printingID))
    }

    public func matches(for query: [Float], k: Int) -> [ArtMatch] {
        // Over-fetch so unlabeled rows don't eat into k.
        index.nearest(to: query, k: k + 8)
            .compactMap { hit in printingIDs[hit.row].map { ArtMatch(printingID: $0, similarity: hit.similarity) } }
            .prefix(k)
            .map { $0 }
    }
}

public struct ArtMatch: Hashable, Sendable {
    public let printingID: String
    public let similarity: Float

    public init(printingID: String, similarity: Float) {
        self.printingID = printingID
        self.similarity = similarity
    }
}

public enum PrintingEmbeddingsError: Error {
    case noEmbeddingRows
}
