import CoreGraphics
import CoreText
import Foundation
import CoreImage
import Testing
import OnePieceKit
@testable import CardVision

/// Draws a synthetic "card" (distinct pattern per seed) so tests need no copyrighted art. With `code`,
/// the card number is printed black-on-white in the bottom-right corner, where `CardOCR` reads it.
func syntheticCard(seed: Int, code: String? = nil, codeScale: CGFloat = 0.05, size: CGSize = CGSize(width: 315, height: 440)) -> CGImage {
    let context = CGContext(
        data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    var rng = SeededGenerator(seed: UInt64(seed))
    context.setFillColor(CGColor(red: .random(in: 0...1, using: &rng), green: .random(in: 0...1, using: &rng), blue: .random(in: 0...1, using: &rng), alpha: 1))
    context.fill(CGRect(origin: .zero, size: size))
    for _ in 0..<40 {
        context.setFillColor(CGColor(red: .random(in: 0...1, using: &rng), green: .random(in: 0...1, using: &rng), blue: .random(in: 0...1, using: &rng), alpha: 1))
        let rect = CGRect(x: .random(in: 0...size.width, using: &rng), y: .random(in: 0...size.height, using: &rng),
                          width: .random(in: 10...120, using: &rng), height: .random(in: 10...120, using: &rng))
        if Bool.random(using: &rng) { context.fillEllipse(in: rect) } else { context.fill(rect) }
    }
    if let code {
        // CGContext's origin is bottom-left, like Vision's normalized coordinates in CardOCR.numberRegion.
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: size.width * 0.55, y: size.height * 0.02, width: size.width * 0.42, height: size.height * 0.09))
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size.height * codeScale, nil)
        let text = NSAttributedString(string: code, attributes: [
            kCTFontAttributeName as NSAttributedString.Key: font,
            kCTForegroundColorAttributeName as NSAttributedString.Key: CGColor(gray: 0, alpha: 1),
        ])
        context.textPosition = CGPoint(x: size.width * 0.58, y: size.height * 0.045)
        CTLineDraw(CTLineCreateWithAttributedString(text), context)
    }
    return context.makeImage()!
}

/// A card composited flat onto a larger dark background, like a photo of a card on a desk.
func photo(of card: CGImage, upsideDown: Bool = false) -> CIImage {
    let w = CGFloat(card.width), h = CGFloat(card.height)
    let background = CIImage(color: CIColor(red: 0.08, green: 0.08, blue: 0.1))
        .cropped(to: CGRect(x: 0, y: 0, width: w * 900 / 315, height: h * 1200 / 440))
    var image = CIImage(cgImage: card)
    if upsideDown { image = image.oriented(.down) }
    let placed = image.transformed(by: CGAffineTransform(translationX: w * 290 / 315, y: h * 380 / 440))
    return placed.composited(over: background)
}

struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

@Suite struct CardVisionTests {
    let context = CIContext()

    @Test func canvasRendersCanonicalSize() throws {
        let image = try #require(CardCanvas.render(CIImage(cgImage: syntheticCard(seed: 1)), context: context))
        #expect(image.width == 630 && image.height == 880)
    }

    @Test func detectorFindsAndRectifiesCard() throws {
        let crop = try #require(try CardDetector(context: context).detectCard(in: photo(of: syntheticCard(seed: 2))))
        #expect(crop.width == 630 && crop.height == 880)
    }

    @Test func featurePrintIsDeterministicAndDiscriminative() throws {
        let engine = EmbeddingEngine()
        let a = try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: syntheticCard(seed: 3)), engine: engine, context: context)
        let a2 = try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: syntheticCard(seed: 3)), engine: engine, context: context)
        let b = try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: syntheticCard(seed: 4)), engine: engine, context: context)
        let index = try EmbeddingIndex(rows: [a, b])
        #expect(index.nearest(to: a2, k: 1).first?.row == 0)
        #expect(engine.backendID == EmbeddingEngine.featurePrintBackendID)
    }

    @Test func recognizerMatchesPhotoToItsReference() throws {
        let engine = EmbeddingEngine()
        let cards = (10..<15).map { syntheticCard(seed: $0) }
        let vectors = try cards.map { try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: $0), engine: engine, context: context) }
        let ids = (10..<15).map { "OP01-0\($0)" }
        let embeddings = try PrintingEmbeddings.build(from: Array(zip(ids, vectors)).map { (printingID: $0.0, vector: $0.1) })
        let catalog = FullCatalog(printingIDs: ids)
        var recognizer = CardRecognizer(engine: engine, matcher: VariantMatcher(catalog: catalog, embeddings: embeddings, source: .bundledIndex), context: context)
        recognizer.ocrEnabled = false

        let result = try #require(try recognizer.recognize(photo: photo(of: cards[2])))
        #expect(result.best?.printingID == ids[2])
        #expect(result.method == .visionOnly)
    }
}

@Suite struct CodeFirstRecognitionTests {
    let context = CIContext()
    let size = CGSize(width: 630, height: 880)

    @Test func readSelectionPrefersCatalogCodesAndKeepsRawReadOtherwise() {
        let valid: (String) -> Bool = { $0 == "OP01-001" }
        // Upright misparses to a non-catalog code; the rotated crop carries the real one.
        let rotatedWins = CardRecognizer.chooseRead(upright: ["OP99-999"], rotated: ["OP01-001"], isValid: valid)
        #expect(rotatedWins?.code == "OP01-001" && rotatedWins?.flipped == true && rotatedWins?.valid == true)
        let uprightWins = CardRecognizer.chooseRead(upright: ["OP01-001"], rotated: ["OP99-999"], isValid: valid)
        #expect(uprightWins?.code == "OP01-001" && uprightWins?.flipped == false)
        // Neither is in the catalog: keep the first upright raw read, flagged invalid.
        let raw = CardRecognizer.chooseRead(upright: ["OP99-999"], rotated: ["OP98-998"], isValid: valid)
        #expect(raw?.code == "OP99-999" && raw?.valid == false)
        let rawRotated = CardRecognizer.chooseRead(upright: [], rotated: ["OP98-998"], isValid: valid)
        #expect(rawRotated?.code == "OP98-998" && rawRotated?.flipped == true && rawRotated?.valid == false)
        #expect(CardRecognizer.chooseRead(upright: [], rotated: [], isValid: valid) == nil)
    }

    @Test func firstCatalogCandidateWins() {
        // A confusion-mapped misread that isn't in the catalog must not beat a later valid candidate.
        let valid: (String) -> Bool = { $0 == "OP05-119" }
        let pick = CardRecognizer.chooseRead(upright: ["OP05-113", "OP05-119"], rotated: [], isValid: valid)
        #expect(pick?.code == "OP05-119" && pick?.valid == true && pick?.flipped == false)
    }

    @Test func numberCropIsEnlarged() throws {
        let card = try #require(CardCanvas.render(CIImage(cgImage: syntheticCard(seed: 90, size: size)), context: context))
        let crop = try #require(CardOCR(context: context).numberCrop(of: card))
        #expect(abs(Double(crop.width) - 0.55 * 630 * 3) < 4)
        #expect(abs(Double(crop.height) - 0.14 * 880 * 3) < 4)
    }

    @Test func readsSmallPrintedCode() throws {
        // ~16 px tall on the canonical card, about the size of a real card's number.
        let card = try #require(CardCanvas.render(
            CIImage(cgImage: syntheticCard(seed: 91, code: "OP05-119", codeScale: 0.012, size: size)), context: context))
        #expect(try CardOCR(context: context).cardIDs(in: card).contains("OP05-119"))
    }

    /// References for `(printingID, seed)` pairs, recognized against a catalog built from the same IDs.
    func makeRecognizer(_ printings: [(id: String, seed: Int)]) throws -> CardRecognizer {
        let engine = EmbeddingEngine()
        let rows = try printings.map { printing in
            (printingID: printing.id,
             vector: try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: syntheticCard(seed: printing.seed, size: size)), engine: engine, context: context))
        }
        let matcher = VariantMatcher(catalog: FullCatalog(printingIDs: printings.map(\.id)),
                                     embeddings: try PrintingEmbeddings.build(from: rows), source: .bundledIndex)
        var recognizer = CardRecognizer(engine: engine, matcher: matcher, context: context)
        recognizer.minimumSimilarity = 0
        return recognizer
    }

    @Test func ocrReadsPrintedCode() throws {
        let card = try #require(CardCanvas.render(CIImage(cgImage: syntheticCard(seed: 20, code: "OP05-119", size: size)), context: context))
        #expect(try CardOCR().cardID(in: card) == "OP05-119")
    }

    @Test func uniqueCodeSkipsVision() throws {
        let recognizer = try makeRecognizer([("OP05-119", 21), ("OP06-118", 22)])
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 21, code: "OP05-119", size: size))))
        #expect(result.method == .ocrUnique && result.groupSize == 1)
        #expect(result.candidates.map(\.printingID) == ["OP05-119"] && result.best?.similarity == nil)
        #expect(result.ocrCardID == "OP05-119")
    }

    @Test func sharedCodeRanksOnlyTheGroup() throws {
        // A distractor from another code is the closest art overall (same seed); it must not appear.
        let recognizer = try makeRecognizer([("OP05-119", 30), ("OP05-119_p1", 31), ("OP06-118", 31)])
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 31, code: "OP05-119", size: size))))
        #expect(result.method == .ocrVision && result.groupSize == 2)
        #expect(result.candidates.map(\.printingID) == ["OP05-119_p1", "OP05-119"])
        #expect(result.best?.similarity != nil)
    }

    @Test func upsideDownCardIsReadRotated() throws {
        let recognizer = try makeRecognizer([("OP05-119", 40), ("OP05-119_p1", 41)])
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 41, code: "OP05-119", size: size), upsideDown: true)))
        #expect(result.flipped && result.method == .ocrVision)
        #expect(result.best?.printingID == "OP05-119_p1")
    }

    @Test func noCodeFallsBackToVision() throws {
        let recognizer = try makeRecognizer([("OP05-119", 50), ("OP06-118", 51)])
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 51, size: size))))
        #expect(result.method == .visionOnly && result.groupSize == 0 && result.ocrCardID == nil)
        #expect(result.best?.printingID == "OP06-118")
    }

    @Test func codeOutsideCatalogFallsBackToVisionAndKeepsTheRead() throws {
        let recognizer = try makeRecognizer([("OP05-119", 60), ("OP06-118", 61)])
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 61, code: "OP09-001", size: size))))
        #expect(result.method == .visionOnly && result.ocrCardID == "OP09-001")
        #expect(result.best?.printingID == "OP06-118")
    }

    @Test func thresholdOnlyAppliesToVisionOnly() throws {
        var strict = try makeRecognizer([("OP05-119", 70), ("OP06-118", 71)])
        strict.minimumSimilarity = 1.01   // nothing passes the threshold
        #expect(try strict.recognize(photo: photo(of: syntheticCard(seed: 72, size: size))) == nil)
        let coded = try strict.recognize(photo: photo(of: syntheticCard(seed: 72, code: "OP05-119", size: size)))
        #expect(coded?.method == .ocrUnique)
    }

    @Test func ocrDisabledForcesVisionOnly() throws {
        var recognizer = try makeRecognizer([("OP05-119", 80), ("OP05-119_p1", 81)])
        recognizer.ocrEnabled = false
        let result = try #require(try recognizer.recognize(photo: photo(of: syntheticCard(seed: 81, code: "OP05-119", size: size))))
        #expect(result.method == .visionOnly && result.ocrCardID == nil)
    }
}
