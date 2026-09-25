import Foundation

/// Pulls a card number such as "OP05-119" out of noisy OCR text.
public enum CardNumberParser {
    // OP05-119, ST01-012, EB01-001, PRB01-001, and promos like P-001.
    nonisolated(unsafe) private static let setPattern = /(OP|ST|EB|PRB)\s?(\d{2})\s?[-–—]\s?(\d{3})/
    nonisolated(unsafe) private static let promoPattern = /\bP\s?[-–—]\s?(\d{3})/

    public static func cardID(in text: String) -> String? {
        let cleaned = normalizeOCR(text.uppercased())
        if let match = cleaned.firstMatch(of: setPattern) {
            return "\(match.1)\(match.2)-\(match.3)"
        }
        if let match = cleaned.firstMatch(of: promoPattern) {
            return "P-\(match.1)"
        }
        return nil
    }

    /// Fixes common OCR confusions in the letter prefix only ("0P05" -> "OP05", "5T01" -> "ST01").
    private static func normalizeOCR(_ text: String) -> String {
        text
            .replacing(/\b0P(?=\s?\d)/, with: "OP")
            .replacing(/\b5T(?=\s?\d)/, with: "ST")
            .replacing(/\bE8(?=\s?\d)/, with: "EB")
    }
}
