import Foundation
import Testing
@testable import OnePieceKit

@Suite struct EmbeddingIndexTests {
    @Test func nearestRanksByCosineSimilarity() throws {
        let index = try EmbeddingIndex(rows: [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
        let hits = index.nearest(to: [10, 0, 0], k: 2)
        #expect(hits.map(\.row) == [0, 2])
        #expect(abs(hits[0].similarity - 1) < 1e-5)
    }

    @Test func rejectsWrongDimensionQuery() throws {
        let index = try EmbeddingIndex(rows: [[1, 0], [0, 1]])
        #expect(index.nearest(to: [1, 0, 0], k: 1).isEmpty)
    }

    @Test func rejectsRaggedShape() {
        #expect(throws: EmbeddingIndexError.self) { try EmbeddingIndex(vectors: [1, 2, 3], dimension: 2) }
    }

    @Test func loadsFlatFloat32File() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID()).f32")
        defer { try? FileManager.default.removeItem(at: url) }
        let floats: [Float] = [1, 0, 0, 1, 0.5, 0.5]
        try floats.withUnsafeBytes { Data($0) }.write(to: url)

        let index = try EmbeddingIndex(contentsOf: url, rowCount: 3)
        #expect(index.dimension == 2)
        #expect(index.nearest(to: [0, 1], k: 1).first?.row == 1)
    }

    @Test func printingEmbeddingsSkipUnlabeledRows() throws {
        let embeddings = PrintingEmbeddings(
            index: try EmbeddingIndex(rows: [[1, 0], [0.99, 0.01], [0, 1]]),
            printingIDs: [nil, "b", "c"])
        #expect(embeddings.matches(for: [1, 0], k: 2).map(\.printingID) == ["b", "c"])
    }

    @Test func printingWithSeveralRowsIsReportedOnceAtBestRow() throws {
        let embeddings = PrintingEmbeddings(
            index: try EmbeddingIndex(rows: [[0.9, 0.1], [1, 0], [0.8, 0.2], [0, 1]]),
            printingIDs: ["a", "b", "a", "c"])
        let matches = embeddings.matches(for: [1, 0], k: 3)
        #expect(matches.map(\.printingID) == ["b", "a", "c"])
        #expect(embeddings.printingCount == 3)
    }

    @Test func loadsWithMetadataAndChecksDimension() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID()).f32")
        defer { try? FileManager.default.removeItem(at: url) }
        let floats: [Float] = [1, 0, 0, 1, 1, 1]
        try floats.withUnsafeBytes { Data($0) }.write(to: url)

        let metadata = EmbeddingIndexMetadata(backend: "test", dimension: 2, rows: ["a", "b", "a"])
        let embeddings = try PrintingEmbeddings.load(from: url, metadata: metadata)
        #expect(embeddings.matches(for: [0, 1], k: 1).first?.printingID == "b")

        let wrong = EmbeddingIndexMetadata(backend: "test", dimension: 4, rows: ["a", "b"])
        #expect(throws: PrintingEmbeddingsError.dimensionMismatch(expected: 4, actual: 3)) {
            try PrintingEmbeddings.load(from: url, metadata: wrong)
        }
    }

    @Test func restrictedSearchOnlyReturnsAllowedPrintings() throws {
        let index = try EmbeddingIndex(rows: [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.8, 0.2, 0]])
        let embeddings = PrintingEmbeddings(index: index, printingIDs: ["A", "B", "C", "B"])
        let hits = embeddings.matches(for: [1, 0, 0], k: 10, restrictedTo: ["B", "C", "Z"])
        #expect(hits.map(\.printingID) == ["B", "C"])   // B once, at its best row; Z has no rows
        #expect(embeddings.matches(for: [1, 0, 0], k: 1).map(\.printingID) == ["A"])
    }
}

@Suite struct CardNumberParserTests {
    @Test(arguments: [
        ("OP05-119 SEC", "OP05-119"),
        ("st01 - 012", "ST01-012"),
        ("0P06-118", "OP06-118"),
        ("EB01–001", "EB01-001"),
        ("PRB01-001", "PRB01-001"),
        ("P-042", "P-042"),
    ])
    func parses(text: String, expected: String) {
        #expect(CardNumberParser.cardID(in: text) == expected)
    }

    @Test func ignoresNoise() {
        #expect(CardNumberParser.cardID(in: "Monkey D. Luffy 6000") == nil)
        #expect(CardNumberParser.cardID(in: "OP05-11") == nil)
    }

    @Test(arguments: [
        ("OP0S-119 GIC 2", "OP05-119"),   // S read for 5
        ("OPO5-1I9", "OP05-119"),         // O for 0, I for 1
        ("OP05 119 SEC", "OP05-119"),     // dash missing
        ("OP05119", "OP05-119"),          // no separator at all
        ("ST0I-0I2", "ST01-012"),
        ("eb0l–00l", "EB01-001"),         // lowercase l
        ("OP06-1B8", "OP06-188"),         // B for 8
        ("P-O42", "P-042"),
    ])
    func parsesOCRConfusions(text: String, expected: String) {
        #expect(CardNumberParser.cardID(in: text) == expected)
    }

    @Test func listsEveryCandidateOnce() {
        #expect(CardNumberParser.cardIDs(in: "OP05-118 OP05-119 OP05-118") == ["OP05-118", "OP05-119"])
        #expect(CardNumberParser.cardIDs(in: "OP01-001 P-042") == ["OP01-001", "P-042"])
        #expect(CardNumberParser.cardIDs(in: "Monkey D. Luffy 6000").isEmpty)
    }

    @Test func promoStillNeedsItsDash() {
        // "P" alone is too common in card text to accept "P 042".
        #expect(CardNumberParser.cardID(in: "P 042") == nil)
    }
}

@Suite struct RankGroupTests {
    let group = ["OP05-119", "OP05-119_p1", "OP05-119_p2"].map { FullCatalogTests.entry($0) }

    @Test func rankGroupOrdersIndexedMembersBySimilarity() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [
            ArtMatch(printingID: "OP05-119_p2", similarity: 0.9), ArtMatch(printingID: "OP05-119", similarity: 0.7),
            ArtMatch(printingID: "OP05-119_p1", similarity: 0.8),
        ])
        #expect(ranked.map(\.printingID) == ["OP05-119_p2", "OP05-119_p1", "OP05-119"])
        #expect(ranked.allSatisfy { $0.cardID == "OP05-119" })
    }

    @Test func rankGroupAppendsUnindexedMembers() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [ArtMatch(printingID: "OP05-119_p1", similarity: 0.8)])
        #expect(ranked.map(\.printingID) == ["OP05-119_p1", "OP05-119", "OP05-119_p2"])
        #expect(ranked.map(\.similarity) == [0.8, nil, nil])
    }

    @Test func rankGroupWithNoIndexedMembers() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [])
        #expect(ranked.map(\.printingID) == group.map(\.printingId))
        #expect(ranked.allSatisfy { $0.similarity == nil })
    }

    @Test func rankGroupIgnoresMatchesOutsideTheGroup() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [ArtMatch(printingID: "OP06-118", similarity: 0.99)])
        #expect(!ranked.map(\.printingID).contains("OP06-118") && ranked.count == 3)
    }

    @Test func methodRawValues() {
        #expect([RecognitionMethod.ocrUnique, .ocrVision, .visionOnly].map(\.rawValue) == ["ocr-unique", "ocr+vision", "vision-only"])
    }
}

@Suite struct CardQuadTests {
    @Test func sensorCornersRotateBackToLandscape() {
        // Full-frame quad in the portrait image (Vision coords, origin bottom-left).
        let quad = CardQuad(topLeft: CGPoint(x: 0, y: 1), topRight: CGPoint(x: 1, y: 1),
                            bottomRight: CGPoint(x: 1, y: 0), bottomLeft: CGPoint(x: 0, y: 0))
        // The portrait image is the sensor image rotated 90° clockwise (`.oriented(.right)`):
        // portrait top-left came from sensor bottom-left, top-right from top-left, and so on.
        #expect(quad.sensorCorners == [CGPoint(x: 0, y: 1), CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1)])
        let point = CardQuad(topLeft: CGPoint(x: 0.25, y: 0.75), topRight: .zero, bottomRight: .zero, bottomLeft: .zero)
        #expect(point.sensorCorners[0] == CGPoint(x: 0.25, y: 0.75))
    }
}
