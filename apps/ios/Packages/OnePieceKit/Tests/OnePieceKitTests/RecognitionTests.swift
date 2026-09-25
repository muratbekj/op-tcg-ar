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
}

@Suite struct CandidateRankerTests {
    let catalog = CardCatalogTests.catalog

    @Test func ocrOnlyReordersNeverDrops() {
        let matches = [
            ArtMatch(printingID: "OP06-118_p0", similarity: 0.83),
            ArtMatch(printingID: "OP05-119_p1", similarity: 0.82),
            ArtMatch(printingID: "OP05-119_p0", similarity: 0.60),
        ]
        let ranked = CandidateRanker.rank(matches, ocrCardID: "OP05-119", catalog: catalog)
        #expect(ranked.map(\.printingID) == ["OP05-119_p1", "OP05-119_p0", "OP06-118_p0"])
    }

    @Test func unknownOCRKeepsArtOrder() {
        let matches = [ArtMatch(printingID: "OP06-118_p0", similarity: 0.9), ArtMatch(printingID: "OP05-119_p0", similarity: 0.5)]
        let ranked = CandidateRanker.rank(matches, ocrCardID: "OP01-001", catalog: catalog)
        #expect(ranked.map(\.printingID) == ["OP06-118_p0", "OP05-119_p0"])
    }

    @Test func needsOCRWhenCloseOrWeak() {
        #expect(CandidateRanker.needsOCR([ArtMatch(printingID: "a", similarity: 0.95), ArtMatch(printingID: "b", similarity: 0.93)]))
        #expect(CandidateRanker.needsOCR([ArtMatch(printingID: "a", similarity: 0.5)]))
        #expect(!CandidateRanker.needsOCR([ArtMatch(printingID: "a", similarity: 0.95), ArtMatch(printingID: "b", similarity: 0.7)]))
    }
}
