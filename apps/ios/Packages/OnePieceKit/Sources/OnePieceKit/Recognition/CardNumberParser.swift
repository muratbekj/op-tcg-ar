import Foundation

/// Pulls card numbers such as "OP05-119" out of noisy OCR text.
///
/// OCR on the tiny printed number confuses letters and digits and sometimes drops the dash, so the
/// digit positions accept the usual look-alikes (O/Q/D→0, I/L→1, Z→2, S→5, G→6, B→8) and set codes
/// may omit the dash. Callers validate candidates against the catalog, which filters out the
/// occasional false positive this tolerance lets through.
public enum CardNumberParser {
    private static let digitLike = "[0-9OQDILZSGB]"
    // OP05-119, ST01-012, EB01-001, PRB01-001 (dash optional), and promos like P-001 (dash required).
    nonisolated(unsafe) private static let setPattern = try! Regex(
        #"(OP|ST|EB|PRB)\s?(\#(digitLike){2})\s?[-–—]?\s?(\#(digitLike){3})(?![0-9])"#)
    nonisolated(unsafe) private static let promoPattern = try! Regex(
        #"\bP\s?[-–—]\s?(\#(digitLike){3})(?![0-9])"#)

    /// The first card number in the text.
    public static func cardID(in text: String) -> String? {
        cardIDs(in: text).first
    }

    /// Every card number in the text: set codes, then promos, each in reading order, no duplicates.
    public static func cardIDs(in text: String) -> [String] {
        let cleaned = normalizePrefix(text.uppercased())
        var ids: [String] = []
        for match in cleaned.matches(of: setPattern) {
            guard let prefix = match.output[1].substring, let set = match.output[2].substring,
                  let number = match.output[3].substring else { continue }
            ids.append("\(prefix)\(digits(set))-\(digits(number))")
        }
        for match in cleaned.matches(of: promoPattern) {
            guard let number = match.output[1].substring else { continue }
            ids.append("P-\(digits(number))")
        }
        var seen = Set<String>()
        return ids.filter { seen.insert($0).inserted }
    }

    /// Fixes common OCR confusions in the letter prefix ("0P05" -> "OP05", "5T01" -> "ST01").
    private static func normalizePrefix(_ text: String) -> String {
        text
            .replacing(/\b0P(?=\s?[0-9OQDILZSGB])/, with: "OP")
            .replacing(/\b5T(?=\s?[0-9OQDILZSGB])/, with: "ST")
            .replacing(/\bE8(?=\s?[0-9OQDILZSGB])/, with: "EB")
    }

    private static func digits(_ text: Substring) -> String {
        String(text.map { character in
            switch character {
            case "O", "Q", "D": "0".first!
            case "I", "L": "1".first!
            case "Z": "2".first!
            case "S": "5".first!
            case "G": "6".first!
            case "B": "8".first!
            default: character
            }
        })
    }
}
