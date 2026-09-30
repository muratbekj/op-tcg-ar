import Foundation

/// One printing from the full OPTCG catalog (`data/cards/catalog.json`, written by `fetch_cards.py`).
/// Unlike `Printing`, it carries no character or variant data: most catalog cards are not in the roster.
public struct CatalogEntry: Codable, Hashable, Identifiable, Sendable {
    public var id: String { printingId }
    public let printingId: String
    public let cardId: String
    public let name: String
    public let set: String
    /// base, parallel, manga, or promo. Kept as a string so new API kinds never break decoding.
    public let kind: String
    public let rarity: String
    public let artUrl: String?

    public init(printingId: String, cardId: String, name: String, set: String, kind: String, rarity: String, artUrl: String?) {
        self.printingId = printingId
        self.cardId = cardId
        self.name = name
        self.set = set
        self.kind = kind
        self.rarity = rarity
        self.artUrl = artUrl
    }
}

/// Every printing that recognition can identify, grouped by card code. Code-first recognition
/// reads the code, then only has to choose among `printings(forCode:)`.
public struct FullCatalog: Sendable {
    public static let fileName = "catalog.json"

    public let entries: [CatalogEntry]
    private let byID: [String: CatalogEntry]
    private let byCode: [String: [CatalogEntry]]

    public init(entries: [CatalogEntry]) {
        var seen = Set<String>()
        let unique = entries.filter { seen.insert($0.printingId).inserted }
        self.entries = unique
        byID = Dictionary(uniqueKeysWithValues: unique.map { ($0.printingId, $0) })
        byCode = Dictionary(grouping: unique, by: \.cardId).mapValues { group in
            group.sorted { ($0.kind == "base" ? 0 : 1, $0.printingId) < ($1.kind == "base" ? 0 : 1, $1.printingId) }
        }
    }

    public static func load(from url: URL) throws -> FullCatalog {
        let data = try Data(contentsOf: url)
        do {
            return FullCatalog(entries: try JSONDecoder().decode([CatalogEntry].self, from: data))
        } catch {
            throw CatalogError.decoding(file: url.lastPathComponent, underlying: error)
        }
    }

    /// Fallback when no `catalog.json` is bundled: only the roster's printings are identifiable.
    public init(roster: CardCatalog) {
        self.init(entries: roster.printings.map { printing in
            CatalogEntry(
                printingId: printing.id, cardId: printing.cardId, name: roster.card(id: printing.cardId)?.name ?? printing.cardId,
                set: String(printing.cardId.split(separator: "-").first ?? ""), kind: printing.kind.rawValue,
                rarity: printing.rarity, artUrl: printing.artUrl)
        })
    }

    /// Fallback for tools that only have an index: codes are derived from the printing IDs.
    public init(printingIDs: some Sequence<String>) {
        self.init(entries: printingIDs.map { id in
            let cardID = Self.cardID(ofPrinting: id)
            return CatalogEntry(printingId: id, cardId: cardID, name: cardID, set: String(cardID.split(separator: "-").first ?? ""),
                                kind: id == cardID ? "base" : "parallel", rarity: "", artUrl: nil)
        })
    }

    public func entry(id: String) -> CatalogEntry? { byID[id] }

    /// Every printing of a card code, base prints first. Empty for codes not in the catalog.
    public func printings(forCode code: String) -> [CatalogEntry] { byCode[code] ?? [] }

    /// API printing IDs are `<cardId>` or `<cardId>_<suffix>` (`OP05-119_p1`, `OP06-118_r1`).
    public static func cardID(ofPrinting id: String) -> String {
        String(id.split(separator: "_", maxSplits: 1).first ?? Substring(id))
    }
}
