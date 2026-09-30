import Foundation

/// An embedding index whose rows are labeled with printing IDs. A printing may own several rows
/// (reference art plus confirmed scans); matches are reported once per printing.
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

    /// Loads an index described by its sidecar metadata (`printings.meta.json`).
    public static func load(from url: URL, metadata: EmbeddingIndexMetadata) throws -> PrintingEmbeddings {
        let index = try EmbeddingIndex(contentsOf: url, rowCount: metadata.rows.count)
        guard index.dimension == metadata.dimension else {
            throw PrintingEmbeddingsError.dimensionMismatch(expected: metadata.dimension, actual: index.dimension)
        }
        return PrintingEmbeddings(index: index, printingIDs: metadata.rows)
    }

    /// Legacy layout without metadata: each printing's `embeddingRow` labels one row.
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

    public var printingCount: Int { Set(printingIDs.compactMap { $0 }).count }

    /// Top-k printings, each at its best-matching row. `allowed` limits the search to those printings
    /// (code-first recognition ranks only the printings that share the card's code).
    public func matches(for query: [Float], k: Int, restrictedTo allowed: Set<String>? = nil) -> [ArtMatch] {
        var seen = Set<String>()
        var result: [ArtMatch] = []
        for hit in index.nearest(to: query, k: index.rowCount) {
            guard let id = printingIDs[hit.row] else { continue }
            if let allowed, !allowed.contains(id) { continue }
            guard seen.insert(id).inserted else { continue }
            result.append(ArtMatch(printingID: id, similarity: hit.similarity))
            if result.count == k { break }
        }
        return result
    }
}

/// Sidecar for `printings.f32`, written by the ML lab. `backend` must equal the device's
/// `EmbeddingEngine.backendID`, or the similarities are meaningless and the index is ignored.
public struct EmbeddingIndexMetadata: Codable, Hashable, Sendable {
    public static let fileName = "printings.meta.json"

    public let backend: String
    public let dimension: Int
    /// Printing ID for each row, in file order. IDs may repeat.
    public let rows: [String]
    /// Applies only to vision-only results (no catalog code was read): a top match below this is
    /// treated as not a card and scanning continues. Chosen per backend
    /// from `evaluate.py`'s rejection curve; `nil` means use `RecognitionDefaults.minimumSimilarity`.
    public let minimumSimilarity: Float?

    public init(backend: String, dimension: Int, rows: [String], minimumSimilarity: Float? = nil) {
        self.backend = backend
        self.dimension = dimension
        self.rows = rows
        self.minimumSimilarity = minimumSimilarity
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

public enum PrintingEmbeddingsError: Error, Equatable {
    case noEmbeddingRows
    case dimensionMismatch(expected: Int, actual: Int)
}
