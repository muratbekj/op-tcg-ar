import CoreGraphics
import CoreImage
import OnePieceKit
import Vision

/// Reads the card number from the bottom-right corner of a canonical card image. The first step
/// of code-first recognition: the code picks the group of printings the embedder chooses among.
public struct CardOCR {
    /// Vision's normalized coordinates, origin bottom-left.
    public static let numberRegion = CGRect(x: 0.45, y: 0.0, width: 0.55, height: 0.14)
    /// The printed number is only ~12–15 px tall on the 630×880 canvas; OCR reads it far more
    /// reliably when the region is cropped and enlarged first.
    public static let upscale: CGFloat = 3

    private let context: CIContext

    public init(context: CIContext = CIContext()) {
        self.context = context
    }

    /// The first card number found.
    public func cardID(in card: CGImage) throws -> String? {
        try cardIDs(in: card).first
    }

    /// Every card number in Vision's top candidates, best observation first, without duplicates.
    /// Callers pick the first one that exists in the catalog.
    public func cardIDs(in card: CGImage) throws -> [String] {
        guard let region = numberCrop(of: card) else { return [] }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: region, orientation: .up, options: [:]).perform([request])

        var ids: [String] = []
        for observation in request.results ?? [] {
            for candidate in observation.topCandidates(3) {
                ids += CardNumberParser.cardIDs(in: candidate.string)
            }
        }
        var seen = Set<String>()
        return ids.filter { seen.insert($0).inserted }
    }

    /// The number region, cropped out of the card and enlarged by `upscale`.
    func numberCrop(of card: CGImage) -> CGImage? {
        let width = CGFloat(card.width), height = CGFloat(card.height)
        // numberRegion uses Vision's bottom-left origin; CGImage cropping uses top-left.
        let rect = CGRect(
            x: Self.numberRegion.minX * width, y: (1 - Self.numberRegion.maxY) * height,
            width: Self.numberRegion.width * width, height: Self.numberRegion.height * height).integral
        guard let cropped = card.cropping(to: rect) else { return nil }
        let scaled = CIImage(cgImage: cropped).applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey: Self.upscale, kCIInputAspectRatioKey: 1.0,
        ])
        return context.createCGImage(scaled, from: scaled.extent.integral)
    }
}
