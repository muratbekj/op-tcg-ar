import Foundation

/// One physical print of a card (base, parallel, manga rare, promo). Recognition matches against
/// printings, not cards, because every printing of a card shares the same card number.
/// Mirrors `data/cards/printings.json`.
public struct Printing: Codable, Hashable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case base, parallel, manga, promo
    }

    public let id: String
    public let cardId: String
    public let kind: Kind
    public let rarity: String
    public let language: String
    public let artUrl: String?
    /// Normalized art window. Full-bleed alt arts use the whole card.
    public let artCrop: ArtCrop?
    /// Row in the bundled `printings.f32` embedding matrix, if this printing has one.
    public let embeddingRow: Int?
    /// `nil` inherits the card's `defaultVariantId`. Set only when the art shows a different form.
    public let variantId: String?

    public init(
        id: String, cardId: String, kind: Kind = .base, rarity: String = "", language: String = "en",
        artUrl: String? = nil, artCrop: ArtCrop? = nil, embeddingRow: Int? = nil, variantId: String? = nil
    ) {
        self.id = id
        self.cardId = cardId
        self.kind = kind
        self.rarity = rarity
        self.language = language
        self.artUrl = artUrl
        self.artCrop = artCrop
        self.embeddingRow = embeddingRow
        self.variantId = variantId
    }
}

/// Normalized `x, y, width, height` rect, top-left origin. Encoded as a 4-element array.
public struct ArtCrop: Codable, Hashable, Sendable {
    public static let fullCard = ArtCrop(x: 0, y: 0, width: 1, height: 1)

    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let values = try container.decode([Double].self)
        guard values.count == 4 else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "artCrop needs 4 values [x, y, w, h], got \(values.count)")
        }
        self.init(x: values[0], y: values[1], width: values[2], height: values[3])
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode([x, y, width, height])
    }
}
