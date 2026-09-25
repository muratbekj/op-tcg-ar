import Foundation

/// In-memory index over cards, printings, and variants, with the
/// Card -> Printing -> Character -> Variant resolution rules.
public struct CardCatalog: Sendable {
    public static let cardsFile = "cards.json"
    public static let printingsFile = "printings.json"
    public static let variantsFile = "variants.json"

    public let cards: [Card]
    public let printings: [Printing]
    public let variants: [CharacterVariant]

    private let cardsByID: [String: Card]
    private let printingsByID: [String: Printing]
    private let printingsByCardID: [String: [Printing]]
    private let variantsByID: [String: CharacterVariant]

    public init(cards: [Card], printings: [Printing], variants: [CharacterVariant]) {
        self.cards = cards
        self.printings = printings
        self.variants = variants
        cardsByID = Dictionary(cards.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        printingsByID = Dictionary(printings.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        printingsByCardID = Dictionary(grouping: printings, by: \.cardId)
        variantsByID = Dictionary(variants.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// Loads the three JSON files from a directory (the repo's `data/cards/` or the app bundle).
    public static func load(from directory: URL) throws -> CardCatalog {
        let decoder = JSONDecoder()
        func read<T: Decodable>(_ file: String) throws -> T {
            let data = try Data(contentsOf: directory.appending(path: file))
            do {
                return try decoder.decode(T.self, from: data)
            } catch {
                throw CatalogError.decoding(file: file, underlying: error)
            }
        }
        return CardCatalog(
            cards: try read(cardsFile),
            printings: try read(printingsFile),
            variants: try read(variantsFile))
    }

    // MARK: Lookups

    public func card(id: String) -> Card? { cardsByID[id] }
    public func printing(id: String) -> Printing? { printingsByID[id] }
    public func variant(id: String) -> CharacterVariant? { variantsByID[id] }

    /// Printings of a card, base prints first.
    public func printings(ofCard cardID: String) -> [Printing] {
        (printingsByCardID[cardID] ?? []).sorted { lhs, rhs in
            (lhs.kind == .base ? 0 : 1, lhs.id) < (rhs.kind == .base ? 0 : 1, rhs.id)
        }
    }

    public func card(for printing: Printing) -> Card? { cardsByID[printing.cardId] }

    /// A printing's override wins; otherwise it inherits its card's default variant.
    public func variantID(for printing: Printing) -> String? {
        printing.variantId ?? cardsByID[printing.cardId]?.defaultVariantId
    }

    public func variant(for printing: Printing) -> CharacterVariant? {
        variantID(for: printing).flatMap { variantsByID[$0] }
    }

    /// Characters derived from cards plus the variants that reference them.
    public var characters: [Character] {
        let variantIDsByCharacter = Dictionary(grouping: variants, by: \.characterId)
            .mapValues { $0.map(\.id).sorted() }
        var names: [String: String] = [:]
        for card in cards where names[card.characterId] == nil {
            names[card.characterId] = card.name
        }
        return Set(names.keys).union(variantIDsByCharacter.keys).sorted().map { id in
            Character(id: id, name: names[id] ?? id, variantIds: variantIDsByCharacter[id] ?? [])
        }
    }

    // MARK: Validation

    /// Referential integrity problems, empty when the data is consistent.
    public func validate() -> [String] {
        var issues: [String] = []
        func duplicates(_ ids: [String]) -> [String] {
            Dictionary(grouping: ids, by: { $0 }).filter { $0.value.count > 1 }.keys.sorted()
        }
        for id in duplicates(cards.map(\.id)) { issues.append("duplicate card id \(id)") }
        for id in duplicates(printings.map(\.id)) { issues.append("duplicate printing id \(id)") }
        for id in duplicates(variants.map(\.id)) { issues.append("duplicate variant id \(id)") }

        for card in cards {
            if let variant = variantsByID[card.defaultVariantId] {
                if variant.characterId != card.characterId {
                    issues.append("card \(card.id) is character \(card.characterId) but default variant \(variant.id) is \(variant.characterId)")
                }
            } else {
                issues.append("card \(card.id) default variant \(card.defaultVariantId) not found")
            }
            if printingsByCardID[card.id, default: []].isEmpty {
                issues.append("card \(card.id) has no printings")
            }
        }
        for printing in printings {
            if cardsByID[printing.cardId] == nil {
                issues.append("printing \(printing.id) references missing card \(printing.cardId)")
            }
            if let override = printing.variantId, variantsByID[override] == nil {
                issues.append("printing \(printing.id) overrides to missing variant \(override)")
            }
            if let crop = printing.artCrop,
               crop.x < 0 || crop.y < 0 || crop.width <= 0 || crop.height <= 0
                || crop.x + crop.width > 1.0001 || crop.y + crop.height > 1.0001 {
                issues.append("printing \(printing.id) artCrop is outside the unit square")
            }
        }
        for id in duplicates(printings.compactMap { $0.embeddingRow.map(String.init) }) {
            issues.append("embedding row \(id) is used by more than one printing")
        }
        for variant in variants where variant.heightMeters <= 0 {
            issues.append("variant \(variant.id) heightMeters must be positive")
        }
        return issues
    }
}

public enum CatalogError: Error, CustomStringConvertible {
    case decoding(file: String, underlying: Error)

    public var description: String {
        switch self {
        case let .decoding(file, underlying): "Could not decode \(file): \(underlying)"
        }
    }
}
