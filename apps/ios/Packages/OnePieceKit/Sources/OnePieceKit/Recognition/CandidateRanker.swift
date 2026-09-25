import Foundation

public struct RecognitionCandidate: Hashable, Identifiable, Sendable {
    public var id: String { printingID }
    public let printingID: String
    public let cardID: String
    public let similarity: Float
    /// True when OCR read this candidate's card number.
    public let matchesOCR: Bool

    public init(printingID: String, cardID: String, similarity: Float, matchesOCR: Bool) {
        self.printingID = printingID
        self.cardID = cardID
        self.similarity = similarity
        self.matchesOCR = matchesOCR
    }
}

/// Combines art similarity (primary) with OCR (narrowing only). The card number is shared by
/// every printing of a card, so OCR can reorder candidates but never picks a printing by itself.
public enum CandidateRanker {
    /// Top two art matches closer than this count as ambiguous, which triggers OCR.
    public static let ambiguityMargin: Float = 0.04
    /// A top match below this is too weak to trust without OCR.
    public static let confidentSimilarity: Float = 0.80
    /// Reject a frame whose best match is below this (Vision feature print). From the 2026-09-25
    /// eval: 2% of non-roster cards accepted, 62% of roster frames kept. A rejected frame just
    /// means scanning continues; a false accept spawns the wrong character.
    public static let defaultMinimumSimilarity: Float = 0.80

    public static func needsOCR(_ matches: [ArtMatch]) -> Bool {
        guard let top = matches.first else { return false }
        if top.similarity < confidentSimilarity { return true }
        guard matches.count > 1 else { return false }
        return top.similarity - matches[1].similarity < ambiguityMargin
    }

    /// Art matches in similarity order, with printings of the OCR'd card number moved to the front.
    /// Non-matching printings stay in the list as alternatives for the "not this one?" picker.
    public static func rank(_ matches: [ArtMatch], ocrCardID: String?, catalog: CardCatalog) -> [RecognitionCandidate] {
        let candidates = matches.compactMap { match -> RecognitionCandidate? in
            guard let printing = catalog.printing(id: match.printingID) else { return nil }
            return RecognitionCandidate(
                printingID: printing.id,
                cardID: printing.cardId,
                similarity: match.similarity,
                matchesOCR: ocrCardID == printing.cardId)
        }
        guard candidates.contains(where: \.matchesOCR) else { return candidates }
        return candidates.filter(\.matchesOCR) + candidates.filter { !$0.matchesOCR }
    }
}
