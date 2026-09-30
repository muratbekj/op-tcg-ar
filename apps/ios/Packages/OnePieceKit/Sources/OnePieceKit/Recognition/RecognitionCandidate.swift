import Foundation

/// How a recognition result was reached. Recorded in results, scan logs, and eval breakdowns.
public enum RecognitionMethod: String, Codable, Sendable {
    /// The card code has exactly one printing: no embedding needed.
    case ocrUnique = "ocr-unique"
    /// The code has several printings; the embedder ranked only those.
    case ocrVision = "ocr+vision"
    /// No usable code: the embedder searched the whole catalog.
    case visionOnly = "vision-only"
}

public struct RecognitionCandidate: Hashable, Identifiable, Sendable {
    public var id: String { printingID }
    public let printingID: String
    public let cardID: String
    /// Cosine similarity to the printing's best reference row; `nil` when no embedding score exists
    /// (the code had a single printing, or the printing has no reference row).
    public let similarity: Float?

    public init(printingID: String, cardID: String, similarity: Float?) {
        self.printingID = printingID
        self.cardID = cardID
        self.similarity = similarity
    }

    /// Every printing of a code group: indexed members by similarity, then members without index rows
    /// in catalog order (base first). Matches outside the group are ignored.
    public static func rankGroup(_ group: [CatalogEntry], matches: [ArtMatch]) -> [RecognitionCandidate] {
        let members = Dictionary(uniqueKeysWithValues: group.map { ($0.printingId, $0) })
        let ranked = matches.sorted { $0.similarity > $1.similarity }.compactMap { match in
            members[match.printingID].map { RecognitionCandidate(printingID: $0.printingId, cardID: $0.cardId, similarity: match.similarity) }
        }
        let rankedIDs = Set(ranked.map(\.printingID))
        let unranked = group.filter { !rankedIDs.contains($0.printingId) }
            .map { RecognitionCandidate(printingID: $0.printingId, cardID: $0.cardId, similarity: nil) }
        return ranked + unranked
    }
}

public enum RecognitionDefaults {
    /// Reject a vision-only frame whose best match is below this (Vision feature print). From the
    /// 2026-09-25 eval: 2% of non-roster cards accepted, 62% of roster frames kept. A rejected frame
    /// just means scanning continues. It never applies when OCR found a catalog code.
    public static let minimumSimilarity: Float = 0.80
}
