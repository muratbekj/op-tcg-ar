import Foundation
import Testing
@testable import OnePieceKit

@Suite struct CardCatalogTests {
    static let catalog = CardCatalog(
        cards: [
            Card(id: "OP05-119", set: "OP05", number: "119", name: "Monkey D. Luffy", kind: .character,
                 characterId: "luffy", defaultVariantId: "luffy_gear4"),
            Card(id: "OP06-118", set: "OP06", number: "118", name: "Roronoa Zoro", kind: .character,
                 characterId: "zoro", defaultVariantId: "zoro_egghead"),
        ],
        printings: [
            Printing(id: "OP05-119_p1", cardId: "OP05-119", kind: .parallel, variantId: "luffy_gear5"),
            Printing(id: "OP05-119_p0", cardId: "OP05-119", kind: .base),
            Printing(id: "OP06-118_p0", cardId: "OP06-118", kind: .base),
        ],
        variants: ["luffy_gear4", "luffy_gear5", "zoro_egghead"].map { id in
            CharacterVariant(
                id: id, characterId: String(id.split(separator: "_")[0]), name: id, modelAsset: "\(id).usdz",
                heightMeters: 0.12, animations: AnimationSet(idle: "idle", hit: "hit", attacks: []))
        })

    @Test func printingInheritsCardDefaultVariant() throws {
        let base = try #require(Self.catalog.printing(id: "OP05-119_p0"))
        #expect(Self.catalog.variant(for: base)?.id == "luffy_gear4")
    }

    @Test func printingOverrideWins() throws {
        let parallel = try #require(Self.catalog.printing(id: "OP05-119_p1"))
        #expect(Self.catalog.variant(for: parallel)?.id == "luffy_gear5")
    }

    @Test func basePrintingsSortFirst() {
        #expect(Self.catalog.printings(ofCard: "OP05-119").map(\.id) == ["OP05-119_p0", "OP05-119_p1"])
    }

    @Test func charactersAreDerived() {
        let luffy = Self.catalog.characters.first { $0.id == "luffy" }
        #expect(luffy?.name == "Monkey D. Luffy")
        #expect(luffy?.variantIds == ["luffy_gear4", "luffy_gear5"])
    }

    @Test func validCatalogHasNoIssues() {
        #expect(Self.catalog.validate().isEmpty)
    }

    @Test func validateCatchesBrokenReferences() {
        let broken = CardCatalog(
            cards: Self.catalog.cards,
            printings: Self.catalog.printings + [Printing(id: "X_p0", cardId: "MISSING", variantId: "nope")],
            variants: Self.catalog.variants)
        let issues = broken.validate()
        #expect(issues.contains("printing X_p0 references missing card MISSING"))
        #expect(issues.contains("printing X_p0 overrides to missing variant nope"))
    }

    @Test func artCropRoundTripsAsArray() throws {
        let json = #"{"id":"a","cardId":"b","kind":"manga","rarity":"SR","language":"en","artCrop":[0.1,0.2,0.3,0.4]}"#
        let printing = try JSONDecoder().decode(Printing.self, from: Data(json.utf8))
        #expect(printing.artCrop == ArtCrop(x: 0.1, y: 0.2, width: 0.3, height: 0.4))
        let encoded = try JSONEncoder().encode(printing.artCrop)
        #expect(String(decoding: encoded, as: UTF8.self) == "[0.1,0.2,0.3,0.4]")
    }

    @Test func attackDisplayName() {
        #expect(Attack(id: "kong_gun", clip: "attack_heavy").displayName == "Kong Gun")
    }
}

/// Guards the real data files in `data/cards/` so a bad hand edit fails `swift test`.
@Suite struct RepoDataTests {
    /// Tests/OnePieceKitTests/<file> -> repo root is seven levels up.
    static let dataDirectory = (0..<7)
        .reduce(URL(filePath: #filePath)) { url, _ in url.deletingLastPathComponent() }
        .appending(path: "data/cards")

    @Test func repoDataLoadsAndValidates() throws {
        let catalog = try CardCatalog.load(from: Self.dataDirectory)
        #expect(!catalog.cards.isEmpty)
        #expect(catalog.validate() == [])
    }

    @Test func everyPrintingResolvesToAVariant() throws {
        let catalog = try CardCatalog.load(from: Self.dataDirectory)
        for printing in catalog.printings {
            #expect(catalog.variant(for: printing) != nil, "\(printing.id) has no variant")
        }
    }
}
