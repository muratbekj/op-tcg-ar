import Accelerate
import Foundation

/// Row-major matrix of L2-normalized embeddings, searched by cosine similarity with vDSP.
/// A few thousand rows fit comfortably in memory and scan in well under a millisecond.
public struct EmbeddingIndex: Sendable {
    public let dimension: Int
    public let rowCount: Int
    private let matrix: [Float]

    public init(vectors: [Float], dimension: Int) throws {
        guard dimension > 0, !vectors.isEmpty, vectors.count % dimension == 0 else {
            throw EmbeddingIndexError.shapeMismatch(floatCount: vectors.count, dimension: dimension)
        }
        let rowCount = vectors.count / dimension
        var normalized = vectors
        normalized.withUnsafeMutableBufferPointer { buffer in
            for row in 0..<rowCount {
                Self.normalize(UnsafeMutableBufferPointer(rebasing: buffer[row * dimension..<(row + 1) * dimension]))
            }
        }
        self.dimension = dimension
        self.rowCount = rowCount
        matrix = normalized
    }

    public init(rows: [[Float]]) throws {
        guard let dimension = rows.first?.count, rows.allSatisfy({ $0.count == dimension }) else {
            throw EmbeddingIndexError.shapeMismatch(floatCount: rows.reduce(0) { $0 + $1.count }, dimension: rows.first?.count ?? 0)
        }
        try self.init(vectors: rows.flatMap { $0 }, dimension: dimension)
    }

    /// Reads a flat little-endian float32 file with `rowCount` rows.
    public init(contentsOf url: URL, rowCount: Int) throws {
        let data = try Data(contentsOf: url)
        let floatCount = data.count / MemoryLayout<Float>.size
        guard rowCount > 0, data.count % MemoryLayout<Float>.size == 0, floatCount % rowCount == 0 else {
            throw EmbeddingIndexError.shapeMismatch(floatCount: floatCount, dimension: rowCount == 0 ? 0 : floatCount / rowCount)
        }
        let vectors = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        try self.init(vectors: vectors, dimension: floatCount / rowCount)
    }

    /// Top-k rows by cosine similarity, most similar first.
    public func nearest(to query: [Float], k: Int) -> [(row: Int, similarity: Float)] {
        guard query.count == dimension, k > 0 else { return [] }
        var q = query
        q.withUnsafeMutableBufferPointer { Self.normalize($0) }

        var scores = [Float](repeating: 0, count: rowCount)
        // scores (rowCount x 1) = matrix (rowCount x dimension) * q (dimension x 1)
        vDSP_mmul(matrix, 1, q, 1, &scores, 1, vDSP_Length(rowCount), 1, vDSP_Length(dimension))

        return scores.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map { (row: $0.offset, similarity: $0.element) }
    }

    private static func normalize(_ vector: UnsafeMutableBufferPointer<Float>) {
        guard let base = vector.baseAddress else { return }
        var sumOfSquares: Float = 0
        vDSP_svesq(base, 1, &sumOfSquares, vDSP_Length(vector.count))
        guard sumOfSquares > 0 else { return }
        var scale = 1 / sumOfSquares.squareRoot()
        vDSP_vsmul(base, 1, &scale, base, 1, vDSP_Length(vector.count))
    }
}

public enum EmbeddingIndexError: Error, Equatable {
    case shapeMismatch(floatCount: Int, dimension: Int)
}
