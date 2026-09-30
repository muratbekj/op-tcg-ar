# Phase 1: Catalog and Code-First Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize any card in the ~4.2k-printing OPTCG catalog by reading its code first, then letting the embedder pick among the printings that share the code, with a vision-only fallback. The eval reports OCR accuracy, within-group accuracy, and end-to-end accuracy by method.

**Architecture:** A new `FullCatalog` (from `data/cards/catalog.json`) maps codes to printings. `CardRecognizer` OCRs the rectified card (upright, then rotated 180°), looks up the code's group, and either returns the single printing (`ocr-unique`), ranks the group with a restricted index search (`ocr+vision`), or falls back to a full-index search (`vision-only`). The `cardvision` CLI emits `method`/`groupSize`, and the Python lab turns them into the new metrics. The app bundles `catalog.json` and compiles against the new types. Its UX changes are Phase 2.

**Tech Stack:** Swift 6 (SwiftPM package `OnePieceKit`, targets `OnePieceKit` / `CardVision` / `CardVisionCLI`, Swift Testing, Vision, Accelerate), SwiftUI app (Xcode project), Python 3 lab under `uv` (pytest, numpy, pillow).

**Spec:** `docs/superpowers/specs/2026-09-29-code-first-recognition-design.md` (this plan implements "Delivery order" item 1 plus the minimum app wiring so the app keeps building. Phases 2–6 get their own plans.)

## Global Constraints

- Method strings are exactly `ocr-unique`, `ocr+vision`, `vision-only`.
- Group of 1 → no embedding step. Group of ≥2 → rank only the group's rows and return every group member. OCR code not in the catalog → treated as no code.
- No code in either orientation → full-catalog vision search, top 5 for the better orientation, rejected below `minimumSimilarity`. The threshold applies **only** to `vision-only`.
- Card scope: any card in the full OPTCG catalog. Roster cards spawn a character. Non-roster UI is Phase 2, so for now the app ignores non-roster results and keeps scanning.
- `catalog.json` fields: `printingId, cardId, name, set, kind, rarity, artUrl`. It's tracked in git (no art, only facts and URLs).
- Never commit card art, scan crops, `.f32` indexes, or models (see `.gitignore`).
- Fully on-device: recognition never uses the network.
- Commits: no `Co-Authored-By` or Claude attribution trailer (user preference).
- The working tree holds unrelated uncommitted user changes (`apps/ios/Info.plist`, `project.pbxproj` DEVELOPMENT_TEAM lines, `data/cards/{cards,printings,roster,variants}.json`, `ml/results/results.csv`). **Stage only the files each task names.** Task 5 edits `project.pbxproj`: stage it with `git add -p` and include only the build-script hunk.
- `xcode-select` points at the Command Line Tools on this Mac, so every `swift` command runs with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` (the Makefile already exports it for `make test`/`make build`).

Shorthand used below:
- `SWIFT_TEST` = `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --package-path apps/ios/Packages/OnePieceKit`
- Python commands run from `ml/`: `cd ml && uv run pytest -q ...`

## Review Focus

1. **OCR reads a real but wrong code** (e.g. `OP05-118` for `OP05-119`). The recognizer confidently returns the wrong group, with no fallback. It must at least be visible: `ocr_accuracy` counts it as wrong, and the row lands in hard cases. Pinned by `test_ocr_misread_counts_against_ocr_accuracy` (Task 6).
2. **Group members without an index row** (art download failed). They must still appear, after the ranked ones, with `similarity == nil`, and a group where *no* member is indexed must still return every member. Pinned by `rankGroupAppendsUnindexedMembers` and `rankGroupWithNoIndexedMembers` (Task 4).
3. **Upside-down card.** Upright OCR fails, rotated OCR succeeds: the result must use the rotated crop and set `flipped = true`. Pinned by `upsideDownCardIsReadRotated` (Task 4).
4. **Catalog missing** (app bundle without `catalog.json`, or CLI run without `--catalog`). Recognition must still work from the roster or the index rows. Pinned by `fullCatalogFromRoster` and `fullCatalogFromPrintingIDs` (Task 2) and the CLI fallback in Task 4.
5. **Existing `results.csv` with the old header.** Appending a run with new columns must keep every old row and value. Pinned by `test_append_result_migrates_old_header` (Task 6).

---

### Task 1: Write `data/cards/catalog.json` from the API

**Files:**
- Modify: `ml/oplab/paths.py:27`
- Modify: `ml/oplab/fetch.py:52-56`
- Create: `ml/tests/test_fetch.py`
- Generate + commit: `data/cards/catalog.json`

**Interfaces:**
- Produces: `paths.FULL_CATALOG == REPO/data/cards/catalog.json`. `fetch.catalog_entries(rows: list[dict]) -> list[dict]` returns entries with keys `printingId, cardId, name, set, kind, rarity, artUrl`, sorted by `printingId`, deduplicated (first wins). Existing readers of `paths.FULL_CATALOG` (`dataset.py:171`, `embeddings.py:52`, `train.py:116`, `evaluate.py:23`) keep working unchanged.

- [ ] **Step 1: Write the failing test**

`ml/tests/test_fetch.py`:
```python
from oplab import paths
from oplab.fetch import catalog_entries


def row(image_id, card_id, name="Luffy", **extra):
    return {"card_image_id": image_id, "card_set_id": card_id, "card_name": name, "set_id": "OP-05",
            "rarity": "SEC", "card_image": f"https://x/{image_id}.jpg", **extra}


def test_catalog_entries_sorted_and_deduplicated():
    rows = [row("OP05-119_p1", "OP05-119", "Monkey.D.Luffy (119) (Alternate Art)"),
            row("OP05-119", "OP05-119", "Monkey.D.Luffy (119)"),
            row("OP05-119", "OP05-119", "duplicate row")]
    entries = catalog_entries(rows)
    assert [e["printingId"] for e in entries] == ["OP05-119", "OP05-119_p1"]
    assert entries[0] == {"printingId": "OP05-119", "cardId": "OP05-119", "name": "Monkey.D.Luffy",
                          "set": "OP-05", "kind": "base", "rarity": "SEC", "artUrl": "https://x/OP05-119.jpg"}
    assert entries[1]["kind"] == "parallel"


def test_catalog_entries_tolerate_missing_rarity_and_art():
    entries = catalog_entries([row("P-001", "P-001", rarity=None, card_image=None)])
    assert entries[0]["rarity"] == "" and entries[0]["artUrl"] is None


def test_full_catalog_lives_with_card_data():
    assert paths.FULL_CATALOG == paths.DATA_CARDS / "catalog.json"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ml && uv run pytest -q tests/test_fetch.py`
Expected: FAIL with `ImportError: cannot import name 'catalog_entries'`.

- [ ] **Step 3: Implement**

`ml/oplab/paths.py`: delete line 27 (`FULL_CATALOG = PROCESSED / "catalog_full.json"`) and add, right after `INDEX_META = ...`:
```python
FULL_CATALOG = DATA_CARDS / "catalog.json"  # every printing in the API (tracked; bundled into the app)
```

`ml/oplab/fetch.py`: add above `def main`:
```python
def catalog_entries(rows: list[dict]) -> list[dict]:
    """Every printing in the API, one entry each, sorted by printing ID so the file diffs cleanly."""
    entries: dict[str, dict] = {}
    for r in rows:
        entries.setdefault(r["card_image_id"], {
            "printingId": r["card_image_id"], "cardId": r["card_set_id"], "name": optcg.base_name(r["card_name"]),
            "set": r["set_id"], "kind": optcg.printing_kind(r), "rarity": r.get("rarity") or "",
            "artUrl": r.get("card_image"),
        })
    return [entries[key] for key in sorted(entries)]
```
Replace the `io.write_json(paths.FULL_CATALOG, [ ... ])` block in `main` with:
```python
    catalog = catalog_entries(rows)
    io.write_json(paths.FULL_CATALOG, catalog)
    print(f"  catalog: {len(catalog)} printings -> {paths.FULL_CATALOG.relative_to(paths.REPO)}")
```
Also update the module docstring's first line to: `"""Build data/cards/{catalog,cards,printings}.json from roster.json + the OPTCG API, and download art.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass (the existing `test_optcg.py`, `test_metrics.py` and `test_synth.py` included).

- [ ] **Step 5: Generate the catalog from the cached API data**

Run: `cd ml && uv run scripts/fetch_cards.py --art none`
Expected output includes `catalog: 4213 printings -> data/cards/catalog.json` (the count may differ slightly if the cache is refreshed). Check: `python3 -c "import json;d=json.load(open('../data/cards/catalog.json'));print(len(d), d[0])"` prints about 4213 and an entry with all seven keys.

This also rewrites `cards.json`/`printings.json` from `roster.json`. Those carry the user's uncommitted Shanks changes, and the regenerated content is the same data. **Do not stage them.**

- [ ] **Step 6: Commit**

```bash
git add ml/oplab/paths.py ml/oplab/fetch.py ml/tests/test_fetch.py data/cards/catalog.json
git commit -m "Write the full printing catalog to data/cards/catalog.json"
```

---

### Task 2: `FullCatalog` in OnePieceKit

**Files:**
- Create: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Catalog/FullCatalog.swift`
- Create: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/FullCatalogTests.swift`
- Modify: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/CardCatalogTests.swift` (the `RepoDataTests` suite, around lines 72-91)

**Interfaces:**
- Consumes: `CardCatalog` (`cards`, `printings`, `card(id:)`), `Printing` (`id`, `cardId`, `kind`, `rarity`, `artUrl`).
- Produces:
  ```swift
  public struct CatalogEntry: Codable, Hashable, Identifiable, Sendable {
      public var id: String { printingId }
      public let printingId: String, cardId: String, name: String, set: String, kind: String, rarity: String
      public let artUrl: String?
  }
  public struct FullCatalog: Sendable {
      public static let fileName = "catalog.json"
      public let entries: [CatalogEntry]
      public init(entries: [CatalogEntry])
      public static func load(from url: URL) throws -> FullCatalog
      public init(roster: CardCatalog)
      public init(printingIDs: some Sequence<String>)
      public func entry(id: String) -> CatalogEntry?
      public func printings(forCode code: String) -> [CatalogEntry]   // base first, then by id
      public static func cardID(ofPrinting id: String) -> String       // "OP05-119_p1" -> "OP05-119"
  }
  ```

- [ ] **Step 1: Write the failing tests**

`Tests/OnePieceKitTests/FullCatalogTests.swift`:
```swift
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
```
(`CardCatalogTests.catalog` is the existing static fixture in `CardCatalogTests.swift`. Read it first; if it's `private`, make it `static let` with internal access.)

Add to `RepoDataTests` in `CardCatalogTests.swift`:
```swift
    @Test func repoCatalogCoversEveryRosterPrinting() throws {
        let roster = try CardCatalog.load(from: Self.dataDirectory)
        let full = try FullCatalog.load(from: Self.dataDirectory.appending(path: FullCatalog.fileName))
        #expect(full.entries.count > 1000)
        for printing in roster.printings {
            #expect(full.entry(id: printing.id) != nil, "\(printing.id) missing from catalog.json")
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter "FullCatalogTests|RepoDataTests"`
Expected: build error, `cannot find 'FullCatalog' in scope`.

- [ ] **Step 3: Implement**

`Sources/OnePieceKit/Catalog/FullCatalog.swift`:
```swift
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$SWIFT_TEST --filter "FullCatalogTests|RepoDataTests"`
Expected: PASS (the repo test reads the `catalog.json` committed in Task 1).

- [ ] **Step 5: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Catalog/FullCatalog.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/FullCatalogTests.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/CardCatalogTests.swift
git commit -m "Add FullCatalog for code-to-printings lookup"
```

---

### Task 3: Restricted index search

**Files:**
- Modify: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/PrintingEmbeddings.swift` (`matches(for:k:)`, around lines 55-66)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift` (the `EmbeddingIndexTests` suite)

**Interfaces:**
- Produces: `PrintingEmbeddings.matches(for query: [Float], k: Int, restrictedTo allowed: Set<String>? = nil) -> [ArtMatch]`. With `allowed`, only those printings are returned, each once at its best row, most similar first. Printings in `allowed` with no rows are absent. Existing callers are unchanged (the default is `nil`).

- [ ] **Step 1: Write the failing tests** (append inside `EmbeddingIndexTests`)

```swift
    @Test func restrictedSearchOnlyReturnsAllowedPrintings() throws {
        let index = try EmbeddingIndex(rows: [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.8, 0.2, 0]])
        let embeddings = PrintingEmbeddings(index: index, printingIDs: ["A", "B", "C", "B"])
        let hits = embeddings.matches(for: [1, 0, 0], k: 10, restrictedTo: ["B", "C", "Z"])
        #expect(hits.map(\.printingID) == ["B", "C"])   // B once, at its best row; Z has no rows
        #expect(embeddings.matches(for: [1, 0, 0], k: 1).map(\.printingID) == ["A"])
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$SWIFT_TEST --filter EmbeddingIndexTests`
Expected: build error, `extra argument 'restrictedTo' in call`.

- [ ] **Step 3: Implement** (replace `matches(for:k:)`)

```swift
    /// Top-k printings, each at its best-matching row. `allowed` limits the search to those printings
    /// (code-first recognition ranks only the printings that share the card's code).
    public func matches(for query: [Float], k: Int, restrictedTo allowed: Set<String>? = nil) -> [ArtMatch] {
        var seen = Set<String>()
        var result: [ArtMatch] = []
        for hit in index.nearest(to: query, k: index.rowCount) {
            guard let id = printingIDs[hit.row] else { continue }
            if let allowed, !allowed.contains(id) { continue }
            guard seen.insert(id).inserted else { continue }
            result.append(ArtMatch(printingID: id, similarity: hit.similarity))
            if result.count == k { break }
        }
        return result
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$SWIFT_TEST --filter EmbeddingIndexTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/PrintingEmbeddings.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift
git commit -m "Support searching the index within a set of printings"
```

---

### Task 4: Code-first `CardRecognizer` and CLI output

This task replaces OCR narrowing with the code-first flow across `OnePieceKit`, `CardVision`, and `CardVisionCLI` together, because they share the changed types. The app is fixed in Task 5 (it isn't built by `swift test`).

**Files:**
- Delete: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/CandidateRanker.swift`
- Create: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/RecognitionCandidate.swift`
- Modify: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/PrintingEmbeddings.swift` (one doc comment that names `CandidateRanker`)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/VariantMatcher.swift` (full rewrite)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardRecognizer.swift` (full rewrite)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardOCR.swift` (doc comment only)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVisionCLI/main.swift` (`Prediction`, `match`, remove `cardID(ofPrinting:)`)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift` (replace the `CandidateRankerTests` suite)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/CardVisionTests/CardVisionTests.swift`

**Interfaces:**
- Consumes: `FullCatalog`, `CatalogEntry` (Task 2), `PrintingEmbeddings.matches(for:k:restrictedTo:)` (Task 3).
- Produces (OnePieceKit):
  ```swift
  public enum RecognitionMethod: String, Codable, Sendable { case ocrUnique = "ocr-unique", ocrVision = "ocr+vision", visionOnly = "vision-only" }
  public struct RecognitionCandidate: Hashable, Identifiable, Sendable {
      public let printingID: String, cardID: String
      public let similarity: Float?          // nil = no embedding score (ocr-unique, or no index row)
      public init(printingID: String, cardID: String, similarity: Float?)
      public static func rankGroup(_ group: [CatalogEntry], matches: [ArtMatch]) -> [RecognitionCandidate]
  }
  public enum RecognitionDefaults { public static let minimumSimilarity: Float = 0.80 }
  ```
- Produces (CardVision):
  ```swift
  public struct RecognitionResult: Sendable {
      public let crop: CGImage, candidates: [RecognitionCandidate], ocrCardID: String?, flipped: Bool
      public let method: RecognitionMethod
      public let groupSize: Int              // printings sharing the read code; 0 for vision-only
      public var best: RecognitionCandidate? { get }
  }
  public struct VariantMatcher: Sendable {
      public init(catalog: FullCatalog, embeddings: PrintingEmbeddings, source: Source)
      public let catalog: FullCatalog
      public var referenceCount: Int { get }
      public func artMatches(for embedding: [Float], k: Int = 5) -> [ArtMatch]
      public func rankGroup(_ group: [CatalogEntry], embedding: [Float]) -> [RecognitionCandidate]
      public func candidates(for matches: [ArtMatch]) -> [RecognitionCandidate]
  }
  ```
  `CardRecognizer`'s public API (`init`, `ocrEnabled`, `candidateCount`, `minimumSimilarity`, `recognize(photo:)`, `recognize(cardImage:)`, `referenceEmbedding`) is unchanged; its default `minimumSimilarity` is now `RecognitionDefaults.minimumSimilarity`.
- CLI `match` prediction JSON line: `{"id", "detected", "flipped", "ocrCardId", "method", "groupSize", "candidates": [{"printingId", "cardId", "similarity"}], "ms", "error"}`. `similarity` may be `null`, and `method` is `null` when nothing was recognized. New optional flag `--catalog <catalog.json>`; without it, the catalog is derived from the index rows.

- [ ] **Step 1: Write the failing pure-logic tests**

In `Tests/OnePieceKitTests/RecognitionTests.swift`, delete the whole `CandidateRankerTests` suite and add:
```swift
@Suite struct RankGroupTests {
    let group = ["OP05-119", "OP05-119_p1", "OP05-119_p2"].map { FullCatalogTests.entry($0) }

    @Test func rankGroupOrdersIndexedMembersBySimilarity() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [
            ArtMatch(printingID: "OP05-119_p2", similarity: 0.9), ArtMatch(printingID: "OP05-119", similarity: 0.7),
            ArtMatch(printingID: "OP05-119_p1", similarity: 0.8),
        ])
        #expect(ranked.map(\.printingID) == ["OP05-119_p2", "OP05-119_p1", "OP05-119"])
        #expect(ranked.allSatisfy { $0.cardID == "OP05-119" })
    }

    @Test func rankGroupAppendsUnindexedMembers() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [ArtMatch(printingID: "OP05-119_p1", similarity: 0.8)])
        #expect(ranked.map(\.printingID) == ["OP05-119_p1", "OP05-119", "OP05-119_p2"])
        #expect(ranked.map(\.similarity) == [0.8, nil, nil])
    }

    @Test func rankGroupWithNoIndexedMembers() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [])
        #expect(ranked.map(\.printingID) == group.map(\.printingId))
        #expect(ranked.allSatisfy { $0.similarity == nil })
    }

    @Test func rankGroupIgnoresMatchesOutsideTheGroup() {
        let ranked = RecognitionCandidate.rankGroup(group, matches: [ArtMatch(printingID: "OP06-118", similarity: 0.99)])
        #expect(!ranked.map(\.printingID).contains("OP06-118") && ranked.count == 3)
    }

    @Test func methodRawValues() {
        #expect([RecognitionMethod.ocrUnique, .ocrVision, .visionOnly].map(\.rawValue) == ["ocr-unique", "ocr+vision", "vision-only"])
    }
}
```

- [ ] **Step 2: Write the failing Vision tests**

In `Tests/CardVisionTests/CardVisionTests.swift`:

1. Add `import CoreText` and `import Foundation` at the top.
2. Replace `syntheticCard` with this version (it keeps the old behavior when `code == nil`):
```swift
/// Draws a synthetic "card" (distinct pattern per seed) so tests need no copyrighted art. With `code`,
/// the card number is printed black-on-white in the bottom-right corner, where `CardOCR` reads it.
func syntheticCard(seed: Int, code: String? = nil, size: CGSize = CGSize(width: 315, height: 440)) -> CGImage {
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
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size.height * 0.05, nil)
        let text = NSAttributedString(string: code, attributes: [
            kCTFontAttributeName as NSAttributedString.Key: font,
            kCTForegroundColorAttributeName as NSAttributedString.Key: CGColor(gray: 0, alpha: 1),
        ])
        context.textPosition = CGPoint(x: size.width * 0.58, y: size.height * 0.045)
        CTLineDraw(CTLineCreateWithAttributedString(text), context)
    }
    return context.makeImage()!
}
```
3. Replace `photo(of:)` so the card keeps the same proportion of the frame at any size:
```swift
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
```
4. Update `recognizerMatchesPhotoToItsReference` for the new `VariantMatcher` init: build the catalog with `let catalog = FullCatalog(printingIDs: ids)` and pass `VariantMatcher(catalog: catalog, embeddings: embeddings, source: .bundledIndex)`. Its `ids` (`OP01-010`…`OP01-014`) are distinct codes and the cards have no printed code, so the path is `vision-only`. Add `#expect(result.method == .visionOnly)` after the existing expectation.
5. Add a code-first suite:
```swift
@Suite struct CodeFirstRecognitionTests {
    let context = CIContext()
    let size = CGSize(width: 630, height: 880)

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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter "RankGroupTests|CardVisionTests|CodeFirstRecognitionTests"`
Expected: build errors (`RecognitionMethod`, `rankGroup`, `method`, `groupSize` not found; `VariantMatcher` init mismatch).

- [ ] **Step 4: Implement the OnePieceKit types**

Delete `Sources/OnePieceKit/Recognition/CandidateRanker.swift`. Create `Sources/OnePieceKit/Recognition/RecognitionCandidate.swift`:
```swift
import Foundation

/// How a recognition result was reached. Recorded in results, scan logs, and eval breakdowns.
public enum RecognitionMethod: String, Codable, Sendable {
    /// The card code has exactly one printing: no embedding needed.
    case ocrUnique = "ocr-unique"
    /// The code has several printings; the embedder ranked only those.
    case ocrVision = "ocr+vision"
    /// No usable code: the embedder searched the whole catalog.
    case visionOnly = "vision-only"
}

public struct RecognitionCandidate: Hashable, Identifiable, Sendable {
    public var id: String { printingID }
    public let printingID: String
    public let cardID: String
    /// Cosine similarity to the printing's best reference row; `nil` when no embedding score exists
    /// (the code had a single printing, or the printing has no reference row).
    public let similarity: Float?

    public init(printingID: String, cardID: String, similarity: Float?) {
        self.printingID = printingID
        self.cardID = cardID
        self.similarity = similarity
    }

    /// Every printing of a code group: indexed members by similarity, then members without index rows
    /// in catalog order (base first). Matches outside the group are ignored.
    public static func rankGroup(_ group: [CatalogEntry], matches: [ArtMatch]) -> [RecognitionCandidate] {
        let members = Dictionary(uniqueKeysWithValues: group.map { ($0.printingId, $0) })
        let ranked = matches.compactMap { match in
            members[match.printingID].map { RecognitionCandidate(printingID: $0.printingId, cardID: $0.cardId, similarity: match.similarity) }
        }
        let rankedIDs = Set(ranked.map(\.printingID))
        let unranked = group.filter { !rankedIDs.contains($0.printingId) }
            .map { RecognitionCandidate(printingID: $0.printingId, cardID: $0.cardId, similarity: nil) }
        return ranked + unranked
    }
}

public enum RecognitionDefaults {
    /// Reject a vision-only frame whose best match is below this (Vision feature print). From the
    /// 2026-09-25 eval: 2% of non-roster cards accepted, 62% of roster frames kept. A rejected frame
    /// just means scanning continues. It never applies when OCR found a catalog code.
    public static let minimumSimilarity: Float = 0.80
}
```
In `PrintingEmbeddings.swift`, change the `EmbeddingIndexMetadata.minimumSimilarity` doc comment's `CandidateRanker.defaultMinimumSimilarity` to `RecognitionDefaults.minimumSimilarity`.

- [ ] **Step 5: Implement `VariantMatcher`** (replace the file)

```swift
import OnePieceKit

/// Art similarity against per-printing reference embeddings, resolved against the full catalog.
public struct VariantMatcher: Sendable {
    public enum Source: Sendable {
        case bundledIndex
        case computedFromCardArt
    }

    public let catalog: FullCatalog
    public let embeddings: PrintingEmbeddings
    /// Where the reference embeddings came from, for the settings/debug screen.
    public let source: Source

    public init(catalog: FullCatalog, embeddings: PrintingEmbeddings, source: Source) {
        self.catalog = catalog
        self.embeddings = embeddings
        self.source = source
    }

    public var referenceCount: Int { embeddings.printingCount }

    /// Top-k printings over the whole index.
    public func artMatches(for embedding: [Float], k: Int = 5) -> [ArtMatch] {
        embeddings.matches(for: embedding, k: k)
    }

    /// Every printing of a code group, ranked by similarity among the group's own rows.
    public func rankGroup(_ group: [CatalogEntry], embedding: [Float]) -> [RecognitionCandidate] {
        let ids = Set(group.map(\.printingId))
        return RecognitionCandidate.rankGroup(group, matches: embeddings.matches(for: embedding, k: ids.count, restrictedTo: ids))
    }

    public func candidates(for matches: [ArtMatch]) -> [RecognitionCandidate] {
        matches.map { match in
            RecognitionCandidate(
                printingID: match.printingID,
                cardID: catalog.entry(id: match.printingID)?.cardId ?? FullCatalog.cardID(ofPrinting: match.printingID),
                similarity: match.similarity)
        }
    }
}
```

- [ ] **Step 6: Implement `CardRecognizer`** (replace the file)

```swift
import CoreImage
import OnePieceKit

public struct RecognitionResult: Sendable {
    public let crop: CGImage
    /// Best first. For `ocrVision`, every printing of the code; for `visionOnly`, the top matches.
    public let candidates: [RecognitionCandidate]
    /// The code OCR read, even when it isn't in the catalog (kept for misread analysis).
    public let ocrCardID: String?
    /// The card was read rotated 180°.
    public let flipped: Bool
    public let method: RecognitionMethod
    /// Printings sharing the read code; 0 for `visionOnly`.
    public let groupSize: Int

    public var best: RecognitionCandidate? { candidates.first }
}

/// The full recognition pipeline, shared by the app and the `cardvision` CLI so offline
/// evaluation measures exactly what runs on the phone.
///
/// Code first: read the card number, then choose among the printings that share it. Falls back to
/// searching the whole index when no catalog code can be read.
public struct CardRecognizer {
    public let engine: EmbeddingEngine
    public let matcher: VariantMatcher
    public var ocrEnabled = true
    /// Candidates returned by the vision-only fallback.
    public var candidateCount = 5
    /// Vision-only best matches below this return `nil` (probably not a card). 0 disables.
    public var minimumSimilarity: Float = RecognitionDefaults.minimumSimilarity

    private let detector: CardDetector
    private let ocr = CardOCR()
    private let context: CIContext

    public init(engine: EmbeddingEngine, matcher: VariantMatcher, context: CIContext = CIContext()) {
        self.engine = engine
        self.matcher = matcher
        self.context = context
        detector = CardDetector(context: context)
    }

    /// Photo or camera frame: find the card first. `nil` when no card-shaped rectangle is found.
    public func recognize(photo: CIImage) throws -> RecognitionResult? {
        guard let card = try detector.detectCard(in: photo) else { return nil }
        return try recognize(canonicalCard: card)
    }

    /// An image that is already just the card (reference art, logged scan crop).
    public func recognize(cardImage: CIImage) throws -> RecognitionResult? {
        guard let card = CardCanvas.render(cardImage, context: context) else { return nil }
        return try recognize(canonicalCard: card)
    }

    private func recognize(canonicalCard upright: CGImage) throws -> RecognitionResult? {
        let rotated = CardCanvas.rotated180(upright, context: context)
        let read = ocrEnabled ? readCode(upright: upright, rotated: rotated) : nil

        if let read {
            let group = matcher.catalog.printings(forCode: read.code)
            if group.count == 1 {
                return RecognitionResult(
                    crop: read.crop, candidates: RecognitionCandidate.rankGroup(group, matches: []),
                    ocrCardID: read.code, flipped: read.flipped, method: .ocrUnique, groupSize: 1)
            }
            if group.count > 1 {
                return RecognitionResult(
                    crop: read.crop, candidates: matcher.rankGroup(group, embedding: try engine.embedding(for: read.crop)),
                    ocrCardID: read.code, flipped: read.flipped, method: .ocrVision, groupSize: group.count)
            }
        }

        // Vision only: the card might be upside down, so keep whichever orientation matches better.
        var best = (crop: upright, matches: matcher.artMatches(for: try engine.embedding(for: upright), k: candidateCount), flipped: false)
        if let rotated {
            let rotatedMatches = matcher.artMatches(for: try engine.embedding(for: rotated), k: candidateCount)
            if (rotatedMatches.first?.similarity ?? -1) > (best.matches.first?.similarity ?? -1) {
                best = (rotated, rotatedMatches, true)
            }
        }
        guard let top = best.matches.first, top.similarity >= minimumSimilarity else { return nil }
        return RecognitionResult(
            crop: best.crop, candidates: matcher.candidates(for: best.matches),
            ocrCardID: read?.code, flipped: best.flipped, method: .visionOnly, groupSize: 0)
    }

    /// The card code from the upright crop, else from the 180°-rotated one.
    private func readCode(upright: CGImage, rotated: CGImage?) -> (crop: CGImage, code: String, flipped: Bool)? {
        if let code = try? ocr.cardID(in: upright) { return (upright, code, false) }
        if let rotated, let code = try? ocr.cardID(in: rotated) { return (rotated, code, true) }
        return nil
    }

    /// Embedding of an image of a whole card, as stored in reference indexes.
    public static func referenceEmbedding(for cardImage: CIImage, engine: EmbeddingEngine, context: CIContext) throws -> [Float] {
        guard let card = CardCanvas.render(cardImage, context: context) else { throw EmbeddingError.noOutput }
        return try engine.embedding(for: card)
    }
}
```
In `CardOCR.swift`, replace the doc comment above `public struct CardOCR` with:
```swift
/// Reads the card number from the bottom-right corner of a canonical card image. The first step
/// of code-first recognition: the code picks the group of printings the embedder chooses among.
```

- [ ] **Step 7: Update the CLI** (`Sources/CardVisionCLI/main.swift`)

1. In the header comment, change the `match` usage line to include `[--catalog catalog.json]`, and add the line:
   `// --catalog is data/cards/catalog.json; without it, codes are derived from the index's printing IDs.`
2. Replace `struct Prediction` with:
```swift
struct Prediction: Encodable {
    struct Candidate: Encodable {
        let printingId: String
        let cardId: String
        let similarity: Float?
    }

    let id: String
    let detected: Bool
    let flipped: Bool
    let ocrCardId: String?
    let method: String?
    let groupSize: Int
    let candidates: [Candidate]
    let ms: Double
    let error: String?
}
```
   Explicit `null`s: Swift's synthesized `Encodable` *omits* nil optionals. Python reads them with `.get`, so omission is fine. Leave the synthesized encoding.
3. Delete the `func cardID(ofPrinting id: String) -> String` helper.
4. In `match`, replace the three lines from `// Recognition only needs printing -> card ...` through `let catalog = CardCatalog(...)` with:
```swift
    let catalog = try args.optional("catalog").map { try FullCatalog.load(from: URL(filePath: $0)) }
        ?? FullCatalog(printingIDs: metadata.rows)
```
5. In the `do` block of the query loop, build the prediction as:
```swift
            prediction = Prediction(
                id: id, detected: result != nil, flipped: result?.flipped ?? false, ocrCardId: result?.ocrCardID,
                method: result?.method.rawValue, groupSize: result?.groupSize ?? 0,
                candidates: (result?.candidates ?? []).map {
                    .init(printingId: $0.printingID, cardId: $0.cardID, similarity: $0.similarity)
                },
                ms: Date.now.timeIntervalSince(start) * 1000, error: nil)
```
   and in the `catch`: `Prediction(id: id, detected: false, flipped: false, ocrCardId: nil, method: nil, groupSize: 0, candidates: [], ms: ..., error: "\(error)")`.
6. Update the comment above `recognizer.minimumSimilarity = ...` to: `// Evaluation wants every vision-only candidate; evaluate.py derives the rejection threshold itself.`

- [ ] **Step 8: Run the tests to verify they pass, and build the CLI**

Run: `$SWIFT_TEST`
Expected: every suite passes, including `CodeFirstRecognitionTests` (real Vision OCR and feature prints).

Run: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product cardvision --package-path apps/ios/Packages/OnePieceKit`
Expected: `Compiling ... Build complete!`

If `ocrReadsPrintedCode` fails because Vision reads the code with extra characters, check `CardNumberParser` handles it (it does a regex search, so surrounding text is fine). If the text isn't found at all, increase the font size factor from `0.05` to `0.06`. Don't change `CardOCR.numberRegion`, which is calibrated for real cards.

- [ ] **Step 9: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources apps/ios/Packages/OnePieceKit/Tests
git commit -m "Recognize code-first: OCR the card number, rank only its printings"
```
(`git add` on those two directories picks up the deleted `CandidateRanker.swift` too; `git status` should show no other staged files.)

---

### Task 5: App compiles against code-first recognition and bundles the catalog

The app isn't covered by `swift test`; the check is `make build`. Behavior change: a recognized printing that isn't in the roster is ignored (scanning continues), until Phase 2 adds the info panel.

**Files:**
- Modify: `apps/ios/OnePieceAR.xcodeproj/project.pbxproj` (the "Copy card data" shell script, around line 151)
- Modify: `apps/ios/OnePieceAR/Services/RecognitionService.swift` (`prepare`, `loadIndex`)
- Modify: `apps/ios/OnePieceAR/App/AppModel.swift` (`bootstrap`, around lines 93-120)
- Modify: `apps/ios/OnePieceAR/Models/ScanRecord.swift`
- Modify: `apps/ios/OnePieceAR/Services/ScanLogger.swift:21-28`
- Modify: `apps/ios/OnePieceAR/Features/Scanner/ExperienceView.swift` (`RecognitionDebugView`, around lines 216-235)
- Modify: `apps/ios/OnePieceAR/Features/Scanner/AlternativesSheet.swift` (`candidateRow`, around lines 49-78)

**Interfaces:**
- Consumes: `FullCatalog` (Task 2), `VariantMatcher(catalog: FullCatalog, …)`, `RecognitionResult.method/groupSize`, `RecognitionCandidate.similarity: Float?` (Task 4).
- Produces: `RecognitionService.prepare(catalog: CardCatalog, fullCatalog: FullCatalog, bundledIndex: URL?, bundledMetadata: URL?, bundledModel: URL?, cardArt: [(printingID: String, image: CGImage)]) -> Bool`. `ScanRecord.Candidate.similarity` becomes `Float?`; `matchesOCR` stays in the JSON, computed as `candidate.cardID == result.ocrCardID` (Phase 2 reworks the record).

- [ ] **Step 1: Confirm the app is broken**

Run: `make build`
Expected: FAIL with errors in `RecognitionService.swift` (`VariantMatcher` init), `ScanLogger.swift` and `AlternativesSheet.swift` (`matchesOCR`), and `ExperienceView.swift` (optional `similarity`).

- [ ] **Step 2: Bundle `catalog.json`**

In `project.pbxproj`, in the `shellScript` string of the card-data build phase, change
`for f in printings.f32 printings.meta.json; do`
to
`for f in printings.f32 printings.meta.json catalog.json; do`
and in the script's first comment line change `(JSON + optional printings.f32)` to `(JSON + optional printings.f32 and catalog.json)`. `catalog.json` is optional: without it the app falls back to the roster.

- [ ] **Step 3: `RecognitionService`**

Change the `prepare` signature to take the full catalog, and use it for the matcher:
```swift
    func prepare(
        catalog: CardCatalog, fullCatalog: FullCatalog, bundledIndex: URL?, bundledMetadata: URL?, bundledModel: URL?,
        cardArt: [(printingID: String, image: CGImage)]
    ) -> Bool {
```
In its body, replace both `VariantMatcher(catalog: catalog, ...)` calls with `VariantMatcher(catalog: fullCatalog, ...)`. `loadIndex` keeps taking the roster `catalog` (the legacy `embeddingRow` path needs `CardCatalog`). Update the doc comment's first sentence to: `Uses the bundled \`printings.f32\` (the full catalog index) when its metadata says it was built with this device's embedding backend; ...`.

- [ ] **Step 4: `AppModel.bootstrap`**

Just before `let bundle = Bundle.main` (i.e. after the `art` array is built), insert:
```swift
        let fullCatalog: FullCatalog
        if let url = Bundle.main.url(forResource: "catalog", withExtension: "json"), let loaded = try? FullCatalog.load(from: url) {
            fullCatalog = loaded
        } else {
            print("AppModel: no catalog.json bundled; recognizing roster printings only")
            fullCatalog = FullCatalog(roster: catalog)
        }
```
and pass `fullCatalog: fullCatalog,` right after `catalog: catalog,` in the `recognition.prepare(...)` call. `consider(_:)` stays as is: its `catalog.printing(id: best.printingID)` guard already ignores non-roster results.

- [ ] **Step 5: `ScanRecord` and `ScanLogger`**

In `ScanRecord.Candidate`, change `let similarity: Float` to:
```swift
        /// `nil` when the printing had no embedding score (single-printing code, or no reference row).
        let similarity: Float?
```
In `ScanLogger.log`, change the `candidates:` argument to:
```swift
            candidates: result.candidates.map {
                .init(printingID: $0.printingID, similarity: $0.similarity, matchesOCR: $0.cardID == result.ocrCardID)
            },
```

- [ ] **Step 6: Debug view and alternatives sheet**

In `ExperienceView.swift`'s `RecognitionDebugView`, replace the `VStack` contents with:
```swift
                Text("OCR: \(result.ocrCardID ?? "–")  \(result.method.rawValue)  n=\(result.groupSize)")
                ForEach(result.candidates.prefix(3)) { candidate in
                    Text("\(candidate.printingID)  \(candidate.similarity.map { String(format: "%.3f", $0) } ?? "–")")
                }
```
In `AlternativesSheet.swift`'s `candidateRow`, replace `if candidate.matchesOCR {` with `if candidate.cardID == model.lastRecognition?.result.ocrCardID {`, and replace the similarity `Text(candidate.similarity, format: ...)` with:
```swift
                if let similarity = candidate.similarity {
                    Text(similarity, format: .percent.precision(.fractionLength(0)))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
```

- [ ] **Step 7: Build**

Run: `make build`
Expected: `** BUILD SUCCEEDED **` (quiet mode prints nothing on success; the exit code is 0).

Run: `make test`
Expected: all package tests pass.

- [ ] **Step 8: Commit**

```bash
git add apps/ios/OnePieceAR/Services/RecognitionService.swift apps/ios/OnePieceAR/App/AppModel.swift \
        apps/ios/OnePieceAR/Models/ScanRecord.swift apps/ios/OnePieceAR/Services/ScanLogger.swift \
        apps/ios/OnePieceAR/Features/Scanner/ExperienceView.swift apps/ios/OnePieceAR/Features/Scanner/AlternativesSheet.swift
git add -p apps/ios/OnePieceAR.xcodeproj/project.pbxproj   # stage ONLY the shellScript hunk, not DEVELOPMENT_TEAM
git commit -m "Wire the app to code-first recognition and bundle catalog.json"
```

---

### Task 6: Eval metrics by method

**Files:**
- Modify: `ml/oplab/metrics.py` (`summarize`)
- Modify: `ml/oplab/evaluate.py` (`RESULT_COLUMNS`, `render_report`, `append_result`, negatives handling in `main`)
- Modify: `ml/tests/test_metrics.py`
- Create: `ml/tests/test_evaluate.py`

**Interfaces:**
- Consumes: CLI prediction lines with `method`, `groupSize`, `candidates[].similarity` possibly absent (Task 4).
- Produces:
  - `metrics.summarize(...)["summary"]` gains `ocr_accuracy` (over detected rows: OCR code == true code, where no read counts as wrong), `within_group` (top-1 over rows with `method == "ocr+vision"` and a correct code), and `ocr_used` (share of detected rows with `method != "vision-only"`). Each row gains `method` and `group_size`; `tags` gains `method`, so `groups["method"]` gives per-method recall automatically.
  - `evaluate.RESULT_COLUMNS` = old columns + `["ocr_accuracy", "within_group"]`. `evaluate.append_result(row)` migrates an old header by rewriting the file.
  - `evaluate.split_negatives(entries: list[dict], indexed: set[str]) -> tuple[list[dict], list[dict]]` returns `(extra_positives, true_negatives)`. A negative whose `tags["actual"]` printing is in the index becomes a positive with that label, because with a full-catalog index it's a card the app *should* identify.

- [ ] **Step 1: Write the failing tests**

Update `prediction` in `ml/tests/test_metrics.py` and add tests:
```python
def prediction(id, *printing_ids, detected=True, ocr=None, method="vision-only", group_size=0):
    return {"id": id, "detected": detected, "ocrCardId": ocr, "ms": 10.0, "method": method, "groupSize": group_size,
            "candidates": [{"printingId": p, "cardId": p.split("_")[0], "similarity": 0.9 - i * 0.1}
                           for i, p in enumerate(printing_ids)]}


def test_method_metrics():
    predictions = [
        prediction("q1", "A_p1", "A", ocr="A", method="ocr+vision", group_size=2),   # right code, right printing
        prediction("q2", "A", "A_p1", ocr="A", method="ocr+vision", group_size=2),   # right code, wrong printing
        prediction("q3", "B", ocr="B", method="ocr-unique", group_size=1),
        prediction("q4", "B", method="vision-only"),                                  # OCR read nothing
    ]
    gt = {**truth("q1", "A_p1"), **truth("q2", "A_p1"), **truth("q3", "B"), **truth("q4", "B")}
    result = metrics.summarize(predictions, gt, lambda p: ATTRS[p])
    s = result["summary"]
    assert s["ocr_accuracy"] == 0.75
    assert s["within_group"] == 0.5
    assert s["ocr_used"] == 0.75
    assert result["groups"]["method"]["ocr+vision"] == {"n": 2, "recall": 0.5}
    assert result["groups"]["method"]["vision-only"]["recall"] == 1.0


def test_ocr_misread_counts_against_ocr_accuracy():
    # OCR read a real but wrong code: confidently wrong group. It must show up, not hide.
    predictions = [prediction("q1", "B", ocr="B", method="ocr-unique", group_size=1)]
    result = metrics.summarize(predictions, truth("q1", "A"), lambda p: ATTRS[p])
    assert result["summary"]["ocr_accuracy"] == 0.0
    assert result["summary"]["within_group"] is None
    assert [c["id"] for c in result["hard_cases"]] == ["q1"]


def test_missing_similarity_is_tolerated():
    p = prediction("q1", "B", ocr="B", method="ocr-unique", group_size=1)
    del p["candidates"][0]["similarity"]   # the CLI omits nil similarities
    result = metrics.summarize([p], truth("q1", "B"), lambda x: ATTRS[x])
    assert result["summary"]["top1"] == 1.0 and result["rows"][0]["similarity"] is None
```
In `test_hard_cases_most_confident_first`, drop `"matchesOCR": False` from the inline candidate (no longer emitted).

`ml/tests/test_evaluate.py`:
```python
import csv

from oplab import dataset, evaluate, paths


def test_split_negatives_turns_indexed_cards_into_positives():
    negatives = [
        {"id": "negative:OP01-077/000_clean.jpg", "path": "x.jpg", "printingId": dataset.NEGATIVE, "cardId": dataset.NEGATIVE,
         "mode": "photo", "tags": {"source": "negative", "condition": "clean", "actual": "OP01-077"}},
        {"id": "negative:OP02-001/000_glare.jpg", "path": "y.jpg", "printingId": dataset.NEGATIVE, "cardId": dataset.NEGATIVE,
         "mode": "photo", "tags": {"source": "negative", "condition": "glare", "actual": "OP02-001"}},
    ]
    positives, true_negatives = evaluate.split_negatives(negatives, indexed={"OP01-077"})
    assert [p["printingId"] for p in positives] == ["OP01-077"]
    assert positives[0]["cardId"] == "OP01-077" and positives[0]["tags"]["source"] == "catalog"
    assert [n["id"] for n in true_negatives] == ["negative:OP02-001/000_glare.jpg"]


def test_append_result_migrates_old_header(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    old_columns = [c for c in evaluate.RESULT_COLUMNS if c not in ("ocr_accuracy", "within_group")]
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({"timestamp": "20260925-132157", "name": "featureprint-baseline", "top1": "0.7704"})
    monkeypatch.setattr(paths, "RESULTS", results)

    evaluate.append_result({"timestamp": "20260929-120000", "name": "code-first", "top1": 0.9, "within_group": 0.8})

    with results.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == evaluate.RESULT_COLUMNS
    assert [r["name"] for r in rows] == ["featureprint-baseline", "code-first"]
    assert rows[0]["top1"] == "0.7704" and rows[0]["within_group"] == ""
    assert rows[1]["within_group"] == "0.8"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ml && uv run pytest -q tests/test_metrics.py tests/test_evaluate.py`
Expected: FAIL (`KeyError: 'within_group'` / `AttributeError: ... 'split_negatives'` / header assertion).

- [ ] **Step 3: Implement `metrics.summarize` changes**

In the per-prediction `rows.append({...})`, add these keys, and replace the `"tags"` and `"ocr_correct"` entries:
```python
            "method": prediction.get("method"),
            "group_size": prediction.get("groupSize", 0),
            "ocr_correct": prediction.get("ocrCardId") == expected["cardId"],
            "tags": {**expected.get("tags", {}), **attributes(expected["printingId"]),
                     "method": prediction.get("method") or "none"},
```
Change `"similarity": top["similarity"] if top else None,` to `"similarity": top.get("similarity") if top else None,`.

In `summary`, replace the `"ocr_used"` and `"ocr_accuracy"` lines with:
```python
        "ocr_used": rate([r["method"] not in (None, "vision-only") for r in detected]),
        "ocr_accuracy": rate([r["ocr_correct"] for r in detected]),
        "within_group": rate([r["top1"] for r in detected if r["method"] == "ocr+vision" and r["ocr_correct"]]),
```
Update the `summarize` docstring's first line to also say: `Rows carry the recognition method; OCR accuracy is over detected rows (no read counts as wrong).`

- [ ] **Step 4: Implement the `evaluate.py` changes**

1. `RESULT_COLUMNS`: append `"ocr_accuracy", "within_group"` at the end of the list.
2. Replace `append_result` with:
```python
def append_result(row: dict) -> None:
    """Appends one run. An older header is migrated by rewriting the file; old rows keep their values."""
    paths.RESULTS.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if paths.RESULTS.exists():
        with paths.RESULTS.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            if reader.fieldnames == RESULT_COLUMNS:
                existing = None  # header is current: just append
    mode = "a" if existing is None else "w"
    with paths.RESULTS.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if existing is not None:
            writer.writeheader()
            for old in existing:
                writer.writerow({k: old.get(k, "") for k in RESULT_COLUMNS})
        writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
```
3. Add above `main`:
```python
def split_negatives(negatives: list[dict], indexed: set[str]) -> tuple[list[dict], list[dict]]:
    """Negatives are photos of cards outside the roster. When the index covers their printing (a
    full-catalog index), they are cards the app should identify, so they become labeled positives.
    Only negatives outside the index remain for the rejection curve."""
    positives, true_negatives = [], []
    for entry in negatives:
        actual = entry["tags"].get("actual")
        if actual in indexed:
            positives.append({**entry, "printingId": actual, "cardId": actual.split("_", 1)[0],
                              "tags": {**entry["tags"], "source": "catalog"}})
        else:
            true_negatives.append(entry)
    return positives, true_negatives
```
4. In `main`, replace
```python
    negatives = [e for e in entries if e["printingId"] == dataset.NEGATIVE]
    in_index = [e for e in entries if e["printingId"] in indexed]
    skipped = len(entries) - len(in_index) - len(negatives)
```
with
```python
    catalog_positives, negatives = split_negatives([e for e in entries if e["printingId"] == dataset.NEGATIVE], indexed)
    in_index = [e for e in entries if e["printingId"] in indexed] + catalog_positives
    skipped = len(entries) - len(in_index) - len(negatives)
```
5. In `main`, pass the catalog to the CLI so codes resolve against the real catalog. Change both `cardvision.match(...)` calls to include `catalog=paths.FULL_CATALOG if paths.FULL_CATALOG.exists() else None`, and in `oplab/cardvision.py` `match` add the parameter `catalog: Path | None = None` and, before `subprocess.run`, `if catalog: command += ["--catalog", str(catalog)]`.
6. Guard the negatives match so it doesn't run with an empty list (`cardvision.match` already returns `[]` for no queries, so no change needed; `rejection_curve` handles empty negatives, and `render_report` skips the section when `rejection["negatives"]` is 0).
7. In `render_report`, replace the `OCR used` / `OCR accuracy when used` rows with:
```python
        f"| OCR read a catalog code | {pct(s['ocr_used'])} |",
        f"| OCR accuracy | {pct(s['ocr_accuracy'])} |",
        f"| Within-group top-1 (right code, ≥2 printings) | {pct(s['within_group'])} |",
```
   and change the `Hardest misses` similarity cell so a missing similarity renders as `–` (it already does via `case['similarity'] is None`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ml/oplab/metrics.py ml/oplab/evaluate.py ml/oplab/cardvision.py ml/tests/test_metrics.py ml/tests/test_evaluate.py
git commit -m "Report OCR, within-group, and per-method accuracy"
```

---

### Task 7: Full-catalog reference index by default

**Files:**
- Modify: `ml/oplab/embeddings.py` (`full_references`, `main`)
- Create: `ml/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `paths.FULL_CATALOG` (Task 1).
- Produces: `embeddings.full_references(with_scans: str = "none") -> list[dict]`: the roster references (API art, the app's own clean art, and optional scans) plus API art for every other catalog printing, deduplicated by image digest. `generate_embeddings.py` defaults to `--scope full` → `data/cards/printings.f32` (+ meta), which the app bundles. `--scope roster` now defaults its output to `ml/datasets/references/roster.f32`.

- [ ] **Step 1: Write the failing test**

`ml/tests/test_embeddings.py`:
```python
from oplab import embeddings, io, paths


def test_full_references_merge_roster_and_catalog(tmp_path, monkeypatch):
    art, app_cards, data = tmp_path / "art", tmp_path / "app", tmp_path / "data"
    for d in (art, app_cards, data):
        d.mkdir()
    (art / "OP05-119.jpg").write_bytes(b"roster-api")
    (art / "OP01-077.jpg").write_bytes(b"catalog-only")
    (art / "OP01-078.jpg").write_bytes(b"roster-api")          # same bytes as a roster image: deduplicated
    (app_cards / "OP05-119.png").write_bytes(b"own-clean-scan")
    io.write_json(data / "printings.json", [{"id": "OP05-119"}])
    io.write_json(data / "catalog.json", [{"printingId": p} for p in ("OP01-077", "OP01-078", "OP05-119", "OP09-999")])
    monkeypatch.setattr(paths, "ART", art)
    monkeypatch.setattr(paths, "APP_CARDS", app_cards)
    monkeypatch.setattr(paths, "DATA_CARDS", data)
    monkeypatch.setattr(paths, "FULL_CATALOG", data / "catalog.json")

    entries = embeddings.full_references()
    assert [(e["printingId"], e["source"]) for e in entries] == [
        ("OP05-119", "api"), ("OP05-119", "app"), ("OP01-077", "api")]   # OP09-999 has no art
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ml && uv run pytest -q tests/test_embeddings.py`
Expected: FAIL (the current `full_references` ignores roster extras and doesn't dedupe; `app` source missing).

- [ ] **Step 3: Implement**

In `embeddings.py`, replace `full_references` with:
```python
def full_references(with_scans: str = "none") -> list[dict]:
    """Roster references first (they may include your own clean art and scans), then the API art of
    every other catalog printing. Identical images are embedded once."""
    entries = roster_references(with_scans)
    seen = {_digest(Path(e["path"])) for e in entries}
    for printing in io.read_json(paths.FULL_CATALOG):
        art = paths.ART / f"{printing['printingId']}.jpg"
        if art.exists() and (digest := _digest(art)) not in seen:
            seen.add(digest)
            entries.append({"printingId": printing["printingId"], "path": str(art), "source": "api"})
    return entries
```
In `main`:
- `--scope` default becomes `"full"`, with help `"full (default): every catalog printing, bundled into the app; roster: roster printings only"`.
- `--with-scans` help: drop `(roster scope)`.
- Replace the scope branch with:
```python
    if args.scope == "roster":
        entries = roster_references(args.with_scans)
        out = args.out or paths.REFERENCES / "roster.f32"
    else:
        entries = full_references(args.with_scans)
        out = args.out or paths.INDEX
    if not entries:
        raise SystemExit("no reference images found; run fetch_cards.py --art all first")
```
- Change the final condition to `if out == paths.INDEX:` (it records roster `embeddingRow`s for either scope).
- Update the module docstring: the full catalog index is the default and goes to `data/cards/printings.f32`; roster scope goes to `ml/datasets/references/roster.f32`; full scope needs `fetch_cards.py --art all` (~1.4 GB).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ml && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ml/oplab/embeddings.py ml/tests/test_embeddings.py
git commit -m "Build the full-catalog reference index by default"
```

---

### Task 8: Docs and the first code-first eval

**Files:**
- Modify: `docs/cv-pipeline.md` (steps 4–6, the embedding contract, the baseline section)
- Modify: `ml/README.md` (walkthrough steps 1, 3, 4)
- Modify: `data/cards/README.md` (file table)
- Modify (generated): `ml/results/results.csv` (**do not commit**: it holds the user's uncommitted rows; report the new row to the user instead)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Update `docs/cv-pipeline.md`**

Replace steps 3–6 of the numbered pipeline with:
```markdown
3. **Read the code.** `CardOCR` reads the bottom-right 55%×14% of the card and `CardNumberParser`
   extracts an `OP05-119`-style code. If the upright crop has none, the 180°-rotated crop is tried,
   and the orientation that produced a code is used from then on.
4. **Look up the group.** `FullCatalog` (`catalog.json`, every printing in the OPTCG API) lists the
   printings that share the code. A code that isn't in the catalog counts as no code.
5. **Pick the printing.**
   - One printing: that's the answer, with no embedding (`ocr-unique`).
   - Several: embed the crop (`VNGenerateImageFeaturePrintRequest` revision 2, or the Core ML
     embedder) and rank only the group's index rows. Every member is returned (`ocr+vision`).
   - No code: embed both orientations, search the whole index, and keep the better orientation's
     top 5 (`vision-only`). Below the index's `minimumSimilarity` (default 0.80), the frame is
     rejected and scanning continues. The threshold never applies when a catalog code was read.
6. **Spawn or show.** A roster printing spawns its character right away, and "Not this one?" lists
   the candidates. Other printings are identified but not spawned yet (the info panel is Phase 2).
```
Renumber the old "Track" step to 7. In the embedding contract, add a bullet: `- The bundled index covers the full catalog (\`generate_embeddings.py\` default scope). \`catalog.json\` is bundled next to it; without it the app recognizes roster printings only.` Under the baseline section, add a line saying the code-first results are in `results.csv` with the `ocr_accuracy` and `within_group` columns.

- [ ] **Step 2: Update `ml/README.md`**

- Step 1: note that `fetch_cards.py` also writes `data/cards/catalog.json` (every printing, tracked in git), and that the full-catalog index needs `--art all` (~1.4 GB, best done on the Mac mini).
- Step 3: `uv run scripts/generate_embeddings.py --min-similarity 0.8` now embeds the whole catalog (about 4.2k rows, a few minutes). `--scope roster` is the old roster-only index.
- Step 4: add to the metric list:
  - **OCR accuracy:** did OCR read the right code (no read counts as wrong)?
  - **Within-group top-1:** given the right code with ≥2 printings, did the embedder pick the right one? This is the number fine-tuning should move.
  - **By method:** accuracy for `ocr-unique`, `ocr+vision`, `vision-only`.
  - Negatives whose printing is in the index are scored as positives ("catalog" source), so the rejection section appears only for a roster-scope index.

- [ ] **Step 3: Update `data/cards/README.md`**

Add a table row: `| \`catalog.json\` | \`fetch_cards.py\` | printing in the whole OPTCG catalog (code, name, set, kind, rarity, art URL): what recognition can identify |`, and change the `printings.f32` row's "One entry per" cell to `reference embedding row (full catalog by default)`.

- [ ] **Step 4: Run the first code-first eval**

```bash
cd ml
uv run scripts/fetch_cards.py --art all          # ~1.4 GB the first time; skips files already present
uv run scripts/generate_embeddings.py --min-similarity 0.8
uv run scripts/prepare_dataset.py build-test
uv run scripts/evaluate.py --name code-first-featureprint
```
Expected: the report prints a metrics table with **OCR accuracy**, **Within-group top-1**, and a **By method** section; `results.csv` gets a row with `index_printings` about 4200. If disk space or time is a problem on this Mac, stop after the docs commit and note that this step runs on the Mac mini.

- [ ] **Step 5: Run the full test suites once more**

Run: `make test && make ml-test && make build`
Expected: all pass, and the build succeeds.

- [ ] **Step 6: Commit the docs only**

```bash
git add docs/cv-pipeline.md ml/README.md data/cards/README.md
git commit -m "Document code-first recognition and the full-catalog index"
```
Report the new `results.csv` row (top1, ocr_accuracy, within_group, and per-method recall from the report) to the user, and leave `results.csv` unstaged. It already carries their uncommitted rows, and they'll decide what to commit.
