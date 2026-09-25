import Foundation

/// One logged scan, written as scan.json next to crop.jpg. This is the device-side half of the
/// learning loop; the Python evaluation reads these files.
nonisolated struct ScanRecord: Codable, Sendable {
    struct Candidate: Codable, Sendable {
        let printingID: String
        let similarity: Float
        let matchesOCR: Bool
    }

    let id: String
    let date: Date
    let ocrCardID: String?
    /// Ranked, best first, as shown in the "not this one?" list.
    let candidates: [Candidate]
    /// What spawned immediately (the top candidate).
    let spawnedPrintingID: String
    /// What the user ended up with; equals `spawnedPrintingID` unless corrected.
    var finalPrintingID: String
    var corrected: Bool
}
