import Foundation
import Testing
@testable import OnePieceKit

@Suite struct FullCatalogTests {
    static func entry(_ id: String, kind: String = "base") -> CatalogEntry {
        CatalogEntry(printingId: id, cardId: FullCatalog.cardID(ofPrinting: id), name: "N", set: "OP-05",
                     kind: kind, rarity: "SEC", artUrl: nil)
    }

    @Test func groupsByCodeBaseFirst() {
        let catalog = FullCatalog(entries: [
            Self.entry("OP05-119_p2", kind: "manga"), Self.entry("OP05-119_p1", kind: "parallel"),
            Self.entry("OP05-119"), Self.entry("OP06-118"),
        ])
        #expect(catalog.printings(forCode: "OP05-119").map(\.printingId) == ["OP05-119", "OP05-119_p1", "OP05-119_p2"])
        #expect(catalog.printings(forCode: "OP01-001").isEmpty)
        #expect(catalog.entry(id: "OP06-118")?.cardId == "OP06-118")
    }

    @Test(arguments: [("OP05-119_p1", "OP05-119"), ("OP06-118_r1", "OP06-118"), ("P-001", "P-001"), ("ST01-012", "ST01-012")])
    func cardIDOfPrinting(id: String, expected: String) {
        #expect(FullCatalog.cardID(ofPrinting: id) == expected)
    }

    @Test func decodesCatalogJSON() throws {
        let json = #"[{"printingId":"OP01-077","cardId":"OP01-077","name":"Perona","set":"OP-01","kind":"base","rarity":"UC","artUrl":null}]"#
        let url = FileManager.default.temporaryDirectory.appending(path: "catalog-\(UUID()).json")
        try Data(json.utf8).write(to: url)
        let catalog = try FullCatalog.load(from: url)
        #expect(catalog.entries.count == 1 && catalog.entry(id: "OP01-077")?.artUrl == nil)
    }

    @Test func fullCatalogFromRoster() {
        let catalog = FullCatalog(roster: CardCatalogTests.catalog)
        #expect(catalog.entries.count == CardCatalogTests.catalog.printings.count)
        let first = CardCatalogTests.catalog.printings[0]
        #expect(catalog.entry(id: first.id)?.cardId == first.cardId)
        #expect(!catalog.printings(forCode: first.cardId).isEmpty)
    }

    @Test func fullCatalogFromPrintingIDs() {
        let catalog = FullCatalog(printingIDs: ["OP05-119_p1", "OP05-119", "OP05-119"])
        #expect(catalog.printings(forCode: "OP05-119").map(\.printingId) == ["OP05-119", "OP05-119_p1"])
    }
}
