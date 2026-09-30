import Foundation
import Testing
@testable import OnePieceKit

@Suite struct ScanRecordTests {
    static func record() -> ScanRecord {
        ScanRecord(
            id: "20260930-120000_abcd1234", date: Date(timeIntervalSince1970: 1_790_000_000), ocrCardID: "OP05-119",
            method: .ocrVision, groupSize: 3,
            candidates: [.init(printingID: "OP05-119_p1", similarity: 0.91, matchesOCR: true),
                         .init(printingID: "OP05-119", similarity: nil, matchesOCR: true)],
            spawnedPrintingID: "OP05-119_p1")
    }

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    @Test func newRecordIsUnlabeled() {
        let record = Self.record()
        #expect(record.label == .unlabeled && !record.corrected && record.finalPrintingID == "OP05-119_p1")
    }

    @Test func resolveComparesAgainstFirstGuess() {
        var record = Self.record()
        record.resolve(to: "OP05-119_p1")
        #expect(record.label == .confirmed && !record.corrected)
        record.resolve(to: "OP05-119")
        #expect(record.label == .corrected && record.corrected && record.finalPrintingID == "OP05-119")
        record.resolve(to: "OP05-119")   // confirming the correction keeps it a correction
        #expect(record.label == .corrected)
        record.resolve(to: "OP05-119_p1")   // back to the first guess
        #expect(record.label == .confirmed && !record.corrected)
    }

    @Test func roundTripsWithLabelStrings() throws {
        var record = Self.record()
        record.resolve(to: "OP05-119")
        let data = try Self.encoder.encode(record)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains(#""label":"corrected""#) && json.contains(#""method":"ocr+vision""#))
        #expect(json.contains(#""corrected":true"#) && json.contains(#""finalPrintingID":"OP05-119""#))
        #expect(try Self.decoder.decode(ScanRecord.self, from: data) == record)
        #expect(try String(data: Self.encoder.encode(ScanLabel.unlabeled), encoding: .utf8) == #""none""#)
    }

    @Test func decodesPhase1Record() throws {
        // Written by the Phase 1 app: no method, groupSize, or label; a nil similarity is omitted.
        let json = #"""
        {"candidates":[{"matchesOCR":true,"printingID":"OP06-118","similarity":0.84},{"matchesOCR":true,"printingID":"OP06-118_p1"}],
         "corrected":true,"date":"2026-09-29T10:00:00Z","finalPrintingID":"OP06-118_p1","id":"20260929-100000_ffff0000",
         "ocrCardID":"OP06-118","spawnedPrintingID":"OP06-118"}
        """#
        var record = try Self.decoder.decode(ScanRecord.self, from: Data(json.utf8))
        #expect(record.method == nil && record.groupSize == 0 && record.label == .corrected)
        #expect(record.candidates[1].similarity == nil)
        record.resolve(to: "OP06-118")
        #expect(record.label == .confirmed)

        let unlabeled = json.replacingOccurrences(of: #""corrected":true"#, with: #""corrected":false"#)
        #expect(try Self.decoder.decode(ScanRecord.self, from: Data(unlabeled.utf8)).label == .unlabeled)
    }

    @Test func labelFromFinalAndFirstGuess() {
        #expect(ScanLabel(finalPrintingID: "A", firstGuess: "A") == .confirmed)
        #expect(ScanLabel(finalPrintingID: "B", firstGuess: "A") == .corrected)
    }
}
