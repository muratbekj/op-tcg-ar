import CoreGraphics
import CoreML
import Foundation
import Vision

/// Turns a canonical card image into an embedding. Two backends:
/// - Vision feature print (revision 2): no training needed, the default.
/// - A Core ML embedding model exported by `ml/scripts/export_coreml.py`.
///
/// Reference embeddings are only comparable with queries from the same backend, which is why every
/// index records `backendID` in its metadata.
public struct EmbeddingEngine {
    public static let featurePrintBackendID = "vision-featureprint-r2"

    private enum Backend {
        case featurePrint
        case coreML(VNCoreMLModel, name: String)
    }

    private let backend: Backend

    /// Vision feature print backend.
    public init() {
        backend = .featurePrint
    }

    /// Core ML backend from a compiled model (`.mlmodelc`).
    public init(compiledModelURL url: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: url, configuration: configuration)
        var name = url.deletingPathExtension().lastPathComponent
        if let version = model.modelDescription.metadata[.versionString] as? String, !version.isEmpty {
            name += "@\(version)"
        }
        backend = .coreML(try VNCoreMLModel(for: model), name: name)
    }

    /// Core ML backend from `.mlpackage`, `.mlmodel`, or `.mlmodelc`, compiling if needed.
    public static func loading(modelAt url: URL) async throws -> EmbeddingEngine {
        if url.pathExtension == "mlmodelc" { return try EmbeddingEngine(compiledModelURL: url) }
        // The backend ID uses the compiled file's name, so make sure it matches the source model's.
        let compiled = try await MLModel.compileModel(at: url)
        let named = compiled.deletingLastPathComponent()
            .appending(path: url.deletingPathExtension().lastPathComponent + ".mlmodelc")
        if compiled.standardizedFileURL != named.standardizedFileURL {
            try? FileManager.default.removeItem(at: named)
            try FileManager.default.moveItem(at: compiled, to: named)
        }
        return try EmbeddingEngine(compiledModelURL: named)
    }

    /// "vision-featureprint-r2", or "coreml:<model name>@<version>" for exported embedders.
    public var backendID: String {
        switch backend {
        case .featurePrint: Self.featurePrintBackendID
        case .coreML(_, let name): "coreml:\(name)"
        }
    }

    public func embedding(for image: CGImage) throws -> [Float] {
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        switch backend {
        case .featurePrint:
            let request = VNGenerateImageFeaturePrintRequest()
            request.revision = VNGenerateImageFeaturePrintRequestRevision2
            request.imageCropAndScaleOption = .scaleFill
            try handler.perform([request])
            guard let observation = request.results?.first else { throw EmbeddingError.noOutput }
            return observation.floatVector
        case .coreML(let model, _):
            let request = VNCoreMLRequest(model: model)
            request.imageCropAndScaleOption = .scaleFill
            try handler.perform([request])
            guard let array = (request.results?.first as? VNCoreMLFeatureValueObservation)?.featureValue.multiArrayValue
            else { throw EmbeddingError.noOutput }
            return array.floatVector
        }
    }
}

public enum EmbeddingError: Error {
    case noOutput
}

extension VNFeaturePrintObservation {
    var floatVector: [Float] {
        switch elementType {
        case .double:
            data.withUnsafeBytes { Array($0.bindMemory(to: Double.self)).map(Float.init) }
        default:
            data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }
}

extension MLMultiArray {
    var floatVector: [Float] {
        (0..<count).map { self[$0].floatValue }
    }
}
