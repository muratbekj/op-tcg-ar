import CardVision
import Foundation
import ImageIO
import OnePieceKit
import UniformTypeIdentifiers

/// Logs every scan (crop + ranked candidates + whether the user corrected it) to
/// Documents/Scans/<id>/. The folder is visible in Finder's device browser and the Files app,
/// which is the export path to the Mac for the learning loop.
actor ScanLogger {
    static var scansDirectory: URL {
        URL.documentsDirectory.appending(path: "Scans", directoryHint: .isDirectory)
    }

    func log(_ result: RecognitionResult, spawnedPrintingID: String) throws -> String {
        let id = "\(Self.timestampFormatter.string(from: .now))_\(UUID().uuidString.prefix(8))"
        let folder = Self.scansDirectory.appending(path: id, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try writeJPEG(result.crop, to: folder.appending(path: "crop.jpg"))

        let record = ScanRecord(
            id: id,
            date: .now,
            ocrCardID: result.ocrCardID,
            candidates: result.candidates.map {
                .init(printingID: $0.printingID, similarity: $0.similarity, matchesOCR: $0.cardID == result.ocrCardID)
            },
            spawnedPrintingID: spawnedPrintingID,
            finalPrintingID: spawnedPrintingID,
            corrected: false)
        try write(record, to: folder)
        return id
    }

    func markCorrected(scanID: String, finalPrintingID: String) throws {
        let folder = Self.scansDirectory.appending(path: scanID, directoryHint: .isDirectory)
        let data = try Data(contentsOf: folder.appending(path: "scan.json"))
        var record = try Self.decoder.decode(ScanRecord.self, from: data)
        record.finalPrintingID = finalPrintingID
        record.corrected = finalPrintingID != record.spawnedPrintingID
        try write(record, to: folder)
    }

    func count() -> Int {
        (try? FileManager.default.contentsOfDirectory(atPath: Self.scansDirectory.path).count) ?? 0
    }

    func deleteAll() throws {
        if FileManager.default.fileExists(atPath: Self.scansDirectory.path) {
            try FileManager.default.removeItem(at: Self.scansDirectory)
        }
    }

    private func write(_ record: ScanRecord, to folder: URL) throws {
        try Self.encoder.encode(record).write(to: folder.appending(path: "scan.json"), options: .atomic)
    }

    private func writeJPEG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
