import CoreGraphics
import Vision

/// Vision feature prints: the "embedding model" that needs no training. Reference embeddings must be
/// produced with the same request revision (see docs/cv-pipeline.md).
nonisolated struct EmbeddingEngine {
    static let revision = VNGenerateImageFeaturePrintRequestRevision2

    func embedding(for image: CGImage) throws -> [Float] {
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = Self.revision
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: image, orientation: .up, options: [:]).perform([request])
        guard let observation = request.results?.first else { throw EmbeddingError.noFeaturePrint }
        return observation.floatVector
    }
}

enum EmbeddingError: Error {
    case noFeaturePrint
}

nonisolated extension VNFeaturePrintObservation {
    var floatVector: [Float] {
        switch elementType {
        case .double:
            data.withUnsafeBytes { Array($0.bindMemory(to: Double.self)).map(Float.init) }
        default:
            data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }
}
