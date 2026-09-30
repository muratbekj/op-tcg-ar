import Foundation

/// How the user resolved a scan. The training set uses only `confirmed` and `corrected` scans.
public enum ScanLabel: String, Codable, Sendable {
    /// The user never confirmed or corrected the pick.
    case unlabeled = "none"
    /// The final printing is the first guess.
    case confirmed
    /// The final printing differs from the first guess.
    case corrected

    public init(finalPrintingID: String, firstGuess: String) {
        self = finalPrintingID == firstGuess ? .confirmed : .corrected
    }
}

/// One logged scan, written as scan.json next to crop.jpg in the app's Documents/Scans. This is the
/// device-side half of the learning loop; the Python lab reads these files, so existing keys keep
/// their names (`spawnedPrintingID` is the first guess even when it only showed the info panel).
public struct ScanRecord: Codable, Sendable, Equatable {
    public struct Candidate: Codable, Sendable, Equatable {
        public let printingID: String
        /// `nil` when the printing had no embedding score (the printing has no reference row).
        public let similarity: Float?
        public let matchesOCR: Bool

        public init(printingID: String, similarity: Float?, matchesOCR: Bool) {
            self.printingID = printingID
            self.similarity = similarity
            self.matchesOCR = matchesOCR
        }
    }

    public let id: String
    public let date: Date
    public let ocrCardID: String?
    /// `nil` in records logged before recognition methods were recorded.
    public let method: RecognitionMethod?
    /// Printings sharing the read code; 0 for vision-only results and older records.
    public let groupSize: Int
    /// Ranked, best first, as shown in the group list.
    public let candidates: [Candidate]
    /// The first guess (spawned, or shown in the info panel).
    public let spawnedPrintingID: String
    /// What the user ended up with; equals `spawnedPrintingID` unless corrected.
    public private(set) var finalPrintingID: String
    /// Kept for older readers: `label == .corrected`.
    public private(set) var corrected: Bool
    public private(set) var label: ScanLabel

    public init(
        id: String, date: Date, ocrCardID: String?, method: RecognitionMethod?, groupSize: Int,
        candidates: [Candidate], spawnedPrintingID: String
    ) {
        self.id = id
        self.date = date
        self.ocrCardID = ocrCardID
        self.method = method
        self.groupSize = groupSize
        self.candidates = candidates
        self.spawnedPrintingID = spawnedPrintingID
        finalPrintingID = spawnedPrintingID
        corrected = false
        label = .unlabeled
    }

    /// The user confirmed `printingID` (the current pick) or chose it instead.
    public mutating func resolve(to printingID: String) {
        finalPrintingID = printingID
        label = ScanLabel(finalPrintingID: printingID, firstGuess: spawnedPrintingID)
        corrected = label == .corrected
    }

    enum CodingKeys: String, CodingKey {
        case id, date, ocrCardID, method, groupSize, candidates, spawnedPrintingID, finalPrintingID, corrected, label
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        date = try container.decode(Date.self, forKey: .date)
        ocrCardID = try container.decodeIfPresent(String.self, forKey: .ocrCardID)
        method = try container.decodeIfPresent(RecognitionMethod.self, forKey: .method)
        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 0
        candidates = try container.decode([Candidate].self, forKey: .candidates)
        spawnedPrintingID = try container.decode(String.self, forKey: .spawnedPrintingID)
        finalPrintingID = try container.decode(String.self, forKey: .finalPrintingID)
        corrected = try container.decode(Bool.self, forKey: .corrected)
        // Older records have no label: a correction is still a correction, anything else is unlabeled.
        label = try container.decodeIfPresent(ScanLabel.self, forKey: .label) ?? (corrected ? .corrected : .unlabeled)
    }
}
