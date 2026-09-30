import CardVision
import CoreImage
import CoreVideo
import OnePieceKit

/// ARKit hands us a CVPixelBuffer on the main actor; recognition reads it on its own actor while
/// the main actor never touches it again.
nonisolated struct PixelBufferBox: @unchecked Sendable {
    let buffer: CVPixelBuffer
}

/// Camera frame -> CardRecognizer (shared with the ML lab's `cardvision` CLI). Runs off the main actor.
actor RecognitionService {
    /// Optional Core ML embedder exported by `ml/scripts/export_coreml.py`. Xcode compiles
    /// `CardEmbedder.mlpackage` in Resources to this.
    static let modelName = "CardEmbedder"

    private var recognizer: CardRecognizer?
    private var summary: String?

    var referenceSummary: String? { summary }

    /// Uses the bundled `printings.f32` (the full catalog index) when its metadata says it was built with this device's
    /// embedding backend; otherwise computes references from bundled card art, so recognition
    /// works for the roster before the ML pipeline has produced anything.
    /// - Returns: `false` when there is nothing to match against.
    func prepare(
        catalog: CardCatalog, fullCatalog: FullCatalog, bundledIndex: URL?, bundledMetadata: URL?, bundledModel: URL?,
        cardArt: [(printingID: String, image: CGImage)]
    ) -> Bool {
        let engine = bundledModel.flatMap { try? EmbeddingEngine(compiledModelURL: $0) } ?? EmbeddingEngine()

        if let bundledIndex, let (embeddings, minimum) = loadIndex(bundledIndex, metadata: bundledMetadata, catalog: catalog, engine: engine) {
            install(engine: engine, matcher: VariantMatcher(catalog: fullCatalog, embeddings: embeddings, source: .bundledIndex),
                    minimumSimilarity: minimum)
            return true
        }

        let context = CIContext()
        let computed = cardArt.compactMap { art in
            (try? CardRecognizer.referenceEmbedding(for: CIImage(cgImage: art.image), engine: engine, context: context))
                .map { (printingID: art.printingID, vector: $0) }
        }
        guard !computed.isEmpty, let embeddings = try? PrintingEmbeddings.build(from: computed) else { return false }
        install(engine: engine, matcher: VariantMatcher(catalog: fullCatalog, embeddings: embeddings, source: .computedFromCardArt),
                minimumSimilarity: nil)
        return true
    }

    private func loadIndex(
        _ url: URL, metadata metadataURL: URL?, catalog: CardCatalog, engine: EmbeddingEngine
    ) -> (PrintingEmbeddings, Float?)? {
        guard let metadataURL else {
            // Legacy layout (embeddingRow only) can't say which backend built it; assume feature prints.
            guard engine.backendID == EmbeddingEngine.featurePrintBackendID,
                  let embeddings = try? PrintingEmbeddings.load(from: url, catalog: catalog) else { return nil }
            return (embeddings, nil)
        }
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONDecoder().decode(EmbeddingIndexMetadata.self, from: data) else { return nil }
        guard metadata.backend == engine.backendID else {
            print("RecognitionService: index built with \(metadata.backend), device uses \(engine.backendID); ignoring it")
            return nil
        }
        guard let embeddings = try? PrintingEmbeddings.load(from: url, metadata: metadata) else { return nil }
        return (embeddings, metadata.minimumSimilarity)
    }

    private func install(engine: EmbeddingEngine, matcher: VariantMatcher, minimumSimilarity: Float?) {
        var recognizer = CardRecognizer(engine: engine, matcher: matcher)
        if let minimumSimilarity { recognizer.minimumSimilarity = minimumSimilarity }
        self.recognizer = recognizer
        let source = switch matcher.source {
        case .bundledIndex: PrintingEmbeddings.fileName
        case .computedFromCardArt: "bundled card art"
        }
        summary = "\(matcher.referenceCount) printings from \(source) (\(engine.backendID))"
    }

    /// - Parameter frame: ARKit's captured image (landscape sensor orientation).
    func attempt(_ frame: PixelBufferBox) throws -> RecognitionAttempt? {
        // Portrait UI: the sensor image must be rotated to match what the user sees.
        try recognizer?.attempt(photo: CIImage(cvPixelBuffer: frame.buffer).oriented(.right))
    }
}
