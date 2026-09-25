import CardVision
import CoreImage
import Foundation
import OnePieceKit

// Mac CLI for the ML lab. Runs the same CardVision code as the phone, so references built here
// and evaluations run here match what the device computes.
//
//   cardvision embed --manifest refs.json --out index.f32 [--meta index.meta.json] [--model M]
//                    [--min-similarity S]
//   cardvision match --index index.f32 [--meta index.meta.json] --queries q.json --out preds.jsonl
//                    [--mode photo|card] [--k 5] [--no-ocr] [--model M] [--min-similarity S]
//
// refs.json:    [{"printingId": "OP05-119_p1", "path": "art/OP05-119_p1.jpg"}, ...]
// queries.json: [{"id": "any-unique-id", "path": "photos/x.jpg"}, ...]
// Relative paths resolve against the JSON file's directory. --model takes .mlpackage/.mlmodel/.mlmodelc;
// without it the Vision feature print backend is used.

struct Usage: Error, CustomStringConvertible {
    let description: String
}

struct Arguments {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []

    init(_ args: ArraySlice<String>) throws {
        var iterator = args.makeIterator()
        while let arg = iterator.next() {
            guard arg.hasPrefix("--") else { throw Usage(description: "unexpected argument \(arg)") }
            let key = String(arg.dropFirst(2))
            if ["no-ocr"].contains(key) {
                flags.insert(key)
            } else if let value = iterator.next() {
                values[key] = value
            } else {
                throw Usage(description: "--\(key) needs a value")
            }
        }
    }

    func required(_ key: String) throws -> String {
        guard let value = values[key] else { throw Usage(description: "missing --\(key)") }
        return value
    }

    func optional(_ key: String) -> String? { values[key] }
    func flag(_ key: String) -> Bool { flags.contains(key) }
}

struct ManifestEntry: Decodable {
    let printingId: String?
    let id: String?
    let path: String
}

struct Prediction: Encodable {
    struct Candidate: Encodable {
        let printingId: String
        let cardId: String
        let similarity: Float
        let matchesOCR: Bool
    }

    let id: String
    let detected: Bool
    let flipped: Bool
    let ocrCardId: String?
    let candidates: [Candidate]
    let ms: Double
    let error: String?
}

func readManifest(_ path: String) throws -> [(entry: ManifestEntry, url: URL)] {
    let url = URL(filePath: path)
    let entries = try JSONDecoder().decode([ManifestEntry].self, from: Data(contentsOf: url))
    let base = url.deletingLastPathComponent()
    return entries.map { entry in
        (entry, entry.path.hasPrefix("/") ? URL(filePath: entry.path) : base.appending(path: entry.path))
    }
}

func loadImage(_ url: URL) throws -> CIImage {
    guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
        throw Usage(description: "cannot read image \(url.path)")
    }
    return image
}

func makeEngine(_ args: Arguments) async throws -> EmbeddingEngine {
    guard let model = args.optional("model") else { return EmbeddingEngine() }
    return try await EmbeddingEngine.loading(modelAt: URL(filePath: model))
}

func defaultMetaPath(for indexPath: String) -> String {
    (indexPath as NSString).deletingPathExtension + ".meta.json"
}

/// Printing IDs from the API are `<cardId>` or `<cardId>_<suffix>`.
func cardID(ofPrinting id: String) -> String {
    String(id.split(separator: "_", maxSplits: 1).first ?? Substring(id))
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func embed(_ args: Arguments) async throws {
    let manifest = try readManifest(try args.required("manifest"))
    let outPath = try args.required("out")
    let engine = try await makeEngine(args)
    let context = CIContext()

    var floats: [Float] = []
    var rows: [String] = []
    var dimension = 0
    for (index, item) in manifest.enumerated() {
        guard let printingID = item.entry.printingId else { throw Usage(description: "manifest entry without printingId") }
        do {
            let vector = try CardRecognizer.referenceEmbedding(for: try loadImage(item.url), engine: engine, context: context)
            if dimension == 0 { dimension = vector.count }
            guard vector.count == dimension else { throw Usage(description: "inconsistent embedding size") }
            floats += vector
            rows.append(printingID)
        } catch {
            log("skip \(item.url.lastPathComponent): \(error)")
        }
        if (index + 1) % 50 == 0 { log("embedded \(index + 1)/\(manifest.count)") }
    }
    guard !rows.isEmpty else { throw Usage(description: "nothing embedded") }

    try floats.withUnsafeBytes { Data($0) }.write(to: URL(filePath: outPath))
    let metadata = EmbeddingIndexMetadata(
        backend: engine.backendID, dimension: dimension, rows: rows,
        minimumSimilarity: args.optional("min-similarity").flatMap(Float.init))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(metadata).write(to: URL(filePath: args.optional("meta") ?? defaultMetaPath(for: outPath)))
    print(#"{"rows": \#(rows.count), "dimension": \#(dimension), "backend": "\#(engine.backendID)"}"#)
}

func match(_ args: Arguments) async throws {
    let indexPath = try args.required("index")
    let metaURL = URL(filePath: args.optional("meta") ?? defaultMetaPath(for: indexPath))
    let metadata = try JSONDecoder().decode(EmbeddingIndexMetadata.self, from: Data(contentsOf: metaURL))
    let engine = try await makeEngine(args)
    guard metadata.backend == engine.backendID else {
        throw Usage(description: "index was built with \(metadata.backend) but the engine is \(engine.backendID)")
    }
    let embeddings = try PrintingEmbeddings.load(from: URL(filePath: indexPath), metadata: metadata)
    // Recognition only needs printing -> card for OCR narrowing, so a minimal catalog suffices.
    let printings = Set(metadata.rows).map { Printing(id: $0, cardId: cardID(ofPrinting: $0)) }
    let catalog = CardCatalog(cards: [], printings: printings, variants: [])

    var recognizer = CardRecognizer(
        engine: engine, matcher: VariantMatcher(catalog: catalog, embeddings: embeddings, source: .bundledIndex))
    recognizer.ocrEnabled = !args.flag("no-ocr")
    // Evaluation wants every candidate; evaluate.py derives the rejection threshold itself.
    recognizer.minimumSimilarity = Float(args.optional("min-similarity") ?? "0") ?? 0
    recognizer.candidateCount = Int(args.optional("k") ?? "5") ?? 5
    let isPhoto = (args.optional("mode") ?? "photo") == "photo"

    let queries = try readManifest(try args.required("queries"))
    let outURL = URL(filePath: try args.required("out"))
    FileManager.default.createFile(atPath: outURL.path, contents: nil)
    let output = try FileHandle(forWritingTo: outURL)
    defer { try? output.close() }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]

    for (index, item) in queries.enumerated() {
        let id = item.entry.id ?? item.entry.path
        let start = Date.now
        var prediction: Prediction
        do {
            let image = try loadImage(item.url)
            let result = isPhoto ? try recognizer.recognize(photo: image) : try recognizer.recognize(cardImage: image)
            prediction = Prediction(
                id: id, detected: result != nil, flipped: result?.flipped ?? false, ocrCardId: result?.ocrCardID,
                candidates: (result?.candidates ?? []).map {
                    .init(printingId: $0.printingID, cardId: $0.cardID, similarity: $0.similarity, matchesOCR: $0.matchesOCR)
                },
                ms: Date.now.timeIntervalSince(start) * 1000, error: nil)
        } catch {
            prediction = Prediction(id: id, detected: false, flipped: false, ocrCardId: nil, candidates: [],
                                    ms: Date.now.timeIntervalSince(start) * 1000, error: "\(error)")
        }
        output.write(try encoder.encode(prediction) + Data("\n".utf8))
        if (index + 1) % 50 == 0 { log("matched \(index + 1)/\(queries.count)") }
    }
}

let arguments = CommandLine.arguments.dropFirst()
do {
    switch arguments.first {
    case "embed": try await embed(try Arguments(arguments.dropFirst()))
    case "match": try await match(try Arguments(arguments.dropFirst()))
    default: throw Usage(description: "usage: cardvision embed|match [options] (see Sources/CardVisionCLI/main.swift)")
    }
} catch {
    log("error: \(error)")
    exit(1)
}
