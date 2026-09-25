import CoreGraphics
import CoreImage
import Testing
import OnePieceKit
@testable import CardVision

/// Draws a synthetic "card" (distinct pattern per seed) so tests need no copyrighted art.
func syntheticCard(seed: Int, size: CGSize = CGSize(width: 315, height: 440)) -> CGImage {
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
    return context.makeImage()!
}

/// A card composited flat onto a larger dark background, like a photo of a card on a desk.
func photo(of card: CGImage) -> CIImage {
    let background = CIImage(color: CIColor(red: 0.08, green: 0.08, blue: 0.1)).cropped(to: CGRect(x: 0, y: 0, width: 900, height: 1200))
    let placed = CIImage(cgImage: card).transformed(by: CGAffineTransform(translationX: 290, y: 380))
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
        let catalog = CardCatalog(cards: [], printings: ids.map { Printing(id: $0, cardId: $0) }, variants: [])
        var recognizer = CardRecognizer(engine: engine, matcher: VariantMatcher(catalog: catalog, embeddings: embeddings, source: .bundledIndex), context: context)
        recognizer.ocrEnabled = false

        let result = try #require(try recognizer.recognize(photo: photo(of: cards[2])))
        #expect(result.best?.printingID == ids[2])
    }
}
