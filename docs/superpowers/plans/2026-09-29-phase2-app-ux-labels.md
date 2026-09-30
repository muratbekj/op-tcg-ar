# Phase 2: App UX, Labels, and OCR Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a scan, the app shows the pick and its code group ("OP05-119 · 4 printings"). The user confirms or corrects it, which logs a training label. Non-roster cards get an info panel instead of a spawn. Group rows show on-demand thumbnails. A debug overlay draws the detected card live. OCR reads the card number far more reliably.

**Architecture:**
- **OCR:** `CardNumberParser` learns OCR digit confusions and returns every candidate code. `CardOCR` crops and upscales the number region. The recognizer takes the first *catalog* code.
- **Detection and summaries:** `CardDetector` also reports the card's quad, and `CardRecognizer.attempt(photo:)` returns it even when nothing matched.
- **Testable logic in OnePieceKit:** `ScanRecord`/`ScanLabel` and the on-disk `ThumbnailStore` move into the package so they can be tested there.
- **App:** `AppModel` tracks the last scan (pick and label) and the identified non-roster card. New SwiftUI views render the strip, the group sheet, the info panel, and the debug overlay.

**Tech Stack:** Swift 6 (SwiftPM package `OnePieceKit` with targets `OnePieceKit` / `CardVision` / `CardVisionCLI`, Swift Testing, Vision, CoreImage), a SwiftUI + ARKit app (Xcode project with file-system-synchronized groups, so adding or deleting a `.swift` file needs no pbxproj edit), and the Python ML lab under `uv` (the Task 2 eval only).

**Spec:** `docs/superpowers/specs/2026-09-29-code-first-recognition-design.md`
- This plan covers "Delivery order" item 2 (App UX and labels).
- It adds the OCR hardening the Phase 1 final review deferred to this phase:
  - `CardNumberParser` rejects `OP0S-119` and codes without a dash.
  - The code is only ~12–15 px tall after the 630×880 resample.
- Phase 1 is merged to `main` (PR #1). Branch: `feature/phase2-app-ux`, from `main` @ `6da178b`.

## Global Constraints

- Method strings are exactly `ocr-unique`, `ocr+vision`, `vision-only`. Label strings are exactly `confirmed`, `corrected`, `none`.
- A confirm tap sets `confirmed`. Choosing another printing sets `corrected` and `finalPrintingID`. Everything else stays `none`.
  - The label compares the final printing with the **first guess**, so confirming after a correction stays `corrected`.
- `scan.json` must stay readable by the Python lab: keep `finalPrintingID`, `spawnedPrintingID`, `corrected`, `candidates`, `ocrCardID`, `id`, `date`.
- `scan.json` files written before this phase (no `method`/`groupSize`/`label`) must still decode, and must be re-labelable.
- Roster printing: the best guess spawns immediately. Non-roster printing: a card-info panel (name, code, set, kind, rarity), no spawn.
- Strip text: `"OP05-119 · 4 printings"` (ocr paths) or `"Matched by art · 5 candidates"` (vision-only). Tapping it opens the group list, which replaces "Not this one?". "None of these" opens the manual picker.
- Thumbnails load on demand from `artUrl` and are cached on disk. Offline, rows show text only. **Recognition itself never uses the network.**
- Debug overlay is off by default: detected quad, OCR read, `method`, and similarity scores, drawn live. It's the existing Settings toggle "Show recognition debug".
- Commits: no `Co-Authored-By` or any Claude attribution trailer (user preference). Stage files explicitly by path.
- Swift commands need `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, and package tests must run with `--no-parallel` (Vision deadlocks in parallel). `make test` / `make build` / `make ml-test` already do both.

Shorthand: `SWIFT_TEST` = `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --no-parallel --package-path apps/ios/Packages/OnePieceKit`

## Review Focus

1. **A confusion-mapped misread that lands on a real code**, or a line with an invalid code before the real one. The recognizer must take the first candidate that is in the catalog, not the first one parsed. Pinned by `firstCatalogCandidateWins` (Task 2).
2. **Offline, or the art URL fails.** The thumbnail returns nil without caching an empty file, and a later call can still succeed. Pinned by `failedFetchIsNotCachedAndRetries` (Task 4).
3. **A `scan.json` written by the Phase 1 app** (no `method`/`groupSize`/`label`). It must decode, derive its label from `corrected`, and accept a new confirm or correction. Pinned by `decodesPhase1Record` (Task 3).
4. **Confirm after a correction, or re-choosing the first guess.** Confirming after a correction stays `corrected`, because the final printing differs from the first guess. Going back to the first guess becomes `confirmed`. Pinned by `resolveComparesAgainstFirstGuess` (Task 3).
5. **Debug quad orientation.** Portrait-image corners must map to the landscape sensor coordinates that `ARFrame.displayTransform` expects. Pinned by `sensorCornersRotateBackToLandscape` (Task 5).

---

### Task 1: OCR-tolerant card number parser

**Files:**
- Modify: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/CardNumberParser.swift` (whole file)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift` (`CardNumberParserTests` suite)

**Interfaces:**
- Produces:
  - `CardNumberParser.cardIDs(in text: String) -> [String]`: every card number, set codes before promos, each in reading order, without duplicates.
  - `CardNumberParser.cardID(in:)` keeps its signature and returns `cardIDs(in:).first`.

- [ ] **Step 1: Write the failing tests** (append inside `CardNumberParserTests`)

```swift
    @Test(arguments: [
        ("OP0S-119 GIC 2", "OP05-119"),   // S read for 5
        ("OPO5-1I9", "OP05-119"),         // O for 0, I for 1
        ("OP05 119 SEC", "OP05-119"),     // dash missing
        ("OP05119", "OP05-119"),          // no separator at all
        ("ST0I-0I2", "ST01-012"),
        ("eb0l–00l", "EB01-001"),         // lowercase l
        ("OP06-1B8", "OP06-188"),         // B for 8
        ("P-O42", "P-042"),
    ])
    func parsesOCRConfusions(text: String, expected: String) {
        #expect(CardNumberParser.cardID(in: text) == expected)
    }

    @Test func listsEveryCandidateOnce() {
        #expect(CardNumberParser.cardIDs(in: "OP05-118 OP05-119 OP05-118") == ["OP05-118", "OP05-119"])
        #expect(CardNumberParser.cardIDs(in: "OP01-001 P-042") == ["OP01-001", "P-042"])
        #expect(CardNumberParser.cardIDs(in: "Monkey D. Luffy 6000").isEmpty)
    }

    @Test func promoStillNeedsItsDash() {
        // "P" alone is too common in card text to accept "P 042".
        #expect(CardNumberParser.cardID(in: "P 042") == nil)
    }
```
The existing `parses` and `ignoresNoise` tests must keep passing unchanged. That includes `"OP05-11"` → nil.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter CardNumberParserTests`
Expected: build error `type 'CardNumberParser' has no member 'cardIDs'`.

- [ ] **Step 3: Implement** (replace the file)

```swift
import Foundation

/// Pulls card numbers such as "OP05-119" out of noisy OCR text.
///
/// OCR on the tiny printed number confuses letters and digits and sometimes drops the dash, so the
/// digit positions accept the usual look-alikes (O/Q/D→0, I/L→1, Z→2, S→5, G→6, B→8) and set codes
/// may omit the dash. Callers validate candidates against the catalog, which filters out the
/// occasional false positive this tolerance lets through.
public enum CardNumberParser {
    private static let digitLike = "[0-9OQDILZSGB]"
    // OP05-119, ST01-012, EB01-001, PRB01-001 (dash optional), and promos like P-001 (dash required).
    nonisolated(unsafe) private static let setPattern = try! Regex(
        #"(OP|ST|EB|PRB)\s?(\#(digitLike){2})\s?[-–—]?\s?(\#(digitLike){3})(?![0-9])"#)
    nonisolated(unsafe) private static let promoPattern = try! Regex(
        #"\bP\s?[-–—]\s?(\#(digitLike){3})(?![0-9])"#)

    /// The first card number in the text.
    public static func cardID(in text: String) -> String? {
        cardIDs(in: text).first
    }

    /// Every card number in the text: set codes, then promos, each in reading order, no duplicates.
    public static func cardIDs(in text: String) -> [String] {
        let cleaned = normalizePrefix(text.uppercased())
        var ids: [String] = []
        for match in cleaned.matches(of: setPattern) {
            guard let prefix = match.output[1].substring, let set = match.output[2].substring,
                  let number = match.output[3].substring else { continue }
            ids.append("\(prefix)\(digits(set))-\(digits(number))")
        }
        for match in cleaned.matches(of: promoPattern) {
            guard let number = match.output[1].substring else { continue }
            ids.append("P-\(digits(number))")
        }
        var seen = Set<String>()
        return ids.filter { seen.insert($0).inserted }
    }

    /// Fixes common OCR confusions in the letter prefix ("0P05" -> "OP05", "5T01" -> "ST01").
    private static func normalizePrefix(_ text: String) -> String {
        text
            .replacing(/\b0P(?=\s?[0-9OQDILZSGB])/, with: "OP")
            .replacing(/\b5T(?=\s?[0-9OQDILZSGB])/, with: "ST")
            .replacing(/\bE8(?=\s?[0-9OQDILZSGB])/, with: "EB")
    }

    private static func digits(_ text: Substring) -> String {
        String(text.map { character -> Character in
            switch character {
            case "O", "Q", "D": "0"
            case "I", "L": "1"
            case "Z": "2"
            case "S": "5"
            case "G": "6"
            case "B": "8"
            default: character
            }
        })
    }
}
```
(`Regex(String)` gives an untyped `Regex<AnyRegexOutput>`, which `match.output[n].substring` reads. The string interpolation keeps the character class in one place.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$SWIFT_TEST --filter CardNumberParserTests`
Expected: all parser tests pass, old and new.

- [ ] **Step 5: Run the full package suite**

Run: `$SWIFT_TEST`
Expected: every suite passes. `CodeFirstRecognitionTests` still reads the printed codes.

- [ ] **Step 6: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/CardNumberParser.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift
git commit -m "Tolerate OCR digit confusions and missing dashes in card numbers"
```

---

### Task 2: Crop-and-upscale OCR, first catalog candidate wins, before/after eval

**Files:**
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardOCR.swift` (whole file)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardRecognizer.swift` (`ocr` property + `init`, `readCode`, `chooseRead`, remove `isValid`)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/CardVisionTests/CardVisionTests.swift` (`syntheticCard`, `readSelectionPrefersCatalogCodesAndKeepsRawReadOtherwise`, new tests)
- Modify + commit: `ml/results/results.csv` (two eval rows)
- Modify: `docs/cv-pipeline.md` (the OCR step text)

**Interfaces:**
- Consumes: `CardNumberParser.cardIDs(in:)` (Task 1).
- Produces:
  - `CardOCR(context: CIContext = CIContext())`
  - `CardOCR.cardIDs(in: CGImage) throws -> [String]`
  - `CardOCR.cardID(in:)` (the first of `cardIDs`)
  - `CardOCR.upscale: CGFloat = 3`
  - `CardOCR.numberCrop(of: CGImage) -> CGImage?` (internal)
  - `CardRecognizer.chooseRead(upright: [String], rotated: [String], isValid: (String) -> Bool) -> (code: String, flipped: Bool, valid: Bool)?` (internal static; the parameters become arrays)

- [ ] **Step 1: Record the "before" eval row**

The Phase 1 full-catalog index, art, and test manifest are already on disk (gitignored). Run:
```bash
cd ml && uv run scripts/evaluate.py --name code-first-featureprint
```
Expected: a report whose table includes "OCR accuracy" (about 43%). `ml/results/results.csv` gets a row, and its header migrates to include `ocr_accuracy,within_group`.
- If the run fails because `data/cards/printings.f32` or `ml/datasets/test/manifest.json` is missing, rebuild them with `uv run scripts/generate_embeddings.py --min-similarity 0.8` and `uv run scripts/prepare_dataset.py build-test`, then rerun.
- Keep the printed OCR accuracy for Step 9.

- [ ] **Step 2: Write the failing tests**

In `CardVisionTests.swift`, give `syntheticCard` a code-size parameter. In its signature, add `codeScale: CGFloat = 0.05` after `code: String? = nil`, and in the drawing code use `size.height * codeScale` for the font size (currently `size.height * 0.05`). Leave everything else unchanged.

Replace `readSelectionPrefersCatalogCodesAndKeepsRawReadOtherwise` with:
```swift
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
```
Add to `CodeFirstRecognitionTests`:
```swift
    @Test func numberCropIsEnlarged() throws {
        let card = try #require(CardCanvas.render(CIImage(cgImage: syntheticCard(seed: 90, size: size)), context: context))
        let crop = try #require(CardOCR(context: context).numberCrop(of: card))
        #expect(abs(Double(crop.width) - 0.55 * 630 * 3) < 4)
        #expect(abs(Double(crop.height) - 0.14 * 880 * 3) < 4)
    }

    @Test func readsSmallPrintedCode() throws {
        // ~16 px tall on the canonical card, about the size of a real card's number.
        let card = try #require(CardCanvas.render(
            CIImage(cgImage: syntheticCard(seed: 91, code: "OP05-119", codeScale: 0.018, size: size)), context: context))
        #expect(try CardOCR(context: context).cardIDs(in: card).contains("OP05-119"))
    }
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter "CodeFirstRecognitionTests"`
Expected: build errors (`chooseRead` argument types, no `numberCrop`, no `CardOCR(context:)`).

To confirm `readsSmallPrintedCode` is a real RED, temporarily stub the new API over the *old* behavior: `init(context:)` stores the context, `cardIDs` returns `[cardID(in:)].compactMap { $0 }` with the old region-of-interest OCR, and `numberCrop` returns nil. Run the test and record the failure in the report. Then replace the stub with Step 4.
- If `readsSmallPrintedCode` *passes* on the old code, lower `codeScale` in that test (0.016, 0.014, …) until the old code fails, and record the value.
- Never go below 0.012. If the old code still reads it at 0.012, keep 0.012 and note in the report that the upscale is justified by the eval (Step 8) instead.

- [ ] **Step 4: Implement `CardOCR`** (replace the file)

```swift
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
```

- [ ] **Step 5: Update `CardRecognizer`**

- Change `private let ocr = CardOCR()` to `private let ocr: CardOCR`. In `init`, after `self.context = context`, add `ocr = CardOCR(context: context)`.
- Delete `private static func isValid(_:_:)`.
- Replace `readCode` and `chooseRead` with:
```swift
    /// The card code from the upright crop, else from the 180°-rotated one. Only a code in the
    /// catalog counts as valid; `inCatalog == false` means the raw read is kept for analysis only.
    private func readCode(upright: CGImage, rotated: CGImage?) -> (crop: CGImage, code: String, flipped: Bool, inCatalog: Bool)? {
        let isValid: (String) -> Bool = { !matcher.catalog.printings(forCode: $0).isEmpty }
        let uprightReads = (try? ocr.cardIDs(in: upright)) ?? []
        var rotatedReads: [String] = []
        if let rotated, !uprightReads.contains(where: isValid) {
            rotatedReads = (try? ocr.cardIDs(in: rotated)) ?? []
        }
        guard let pick = Self.chooseRead(upright: uprightReads, rotated: rotatedReads, isValid: isValid) else { return nil }
        return (pick.flipped ? (rotated ?? upright) : upright, pick.code, pick.flipped, pick.valid)
    }

    /// Picks the read to use: the first valid upright candidate, else the first valid rotated one,
    /// else the first raw read (upright first, else rotated) flagged invalid.
    static func chooseRead(upright: [String], rotated: [String], isValid: (String) -> Bool)
        -> (code: String, flipped: Bool, valid: Bool)? {
        if let code = upright.first(where: isValid) { return (code, false, true) }
        if let code = rotated.first(where: isValid) { return (code, true, true) }
        if let code = upright.first { return (code, false, false) }
        if let code = rotated.first { return (code, true, false) }
        return nil
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$SWIFT_TEST`
Expected: every suite passes, including `readsSmallPrintedCode`, `numberCropIsEnlarged`, `firstCatalogCandidateWins`, and all existing `CodeFirstRecognitionTests`.

- [ ] **Step 7: Build the CLI**

Run: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product cardvision --package-path apps/ios/Packages/OnePieceKit`
Expected: `Build complete!`

- [ ] **Step 8: Record the "after" eval row**

Run: `cd ml && uv run scripts/evaluate.py --name ocr-hardened`
Expected: the report's "OCR accuracy" is clearly higher than in Step 1. Copy both reports' main tables and "By method" tables into your task report.
- If OCR accuracy did **not** improve, stop and report DONE_WITH_CONCERNS with both tables. Do not tune further in this task.

- [ ] **Step 9: Document**

In `docs/cv-pipeline.md`, step 3 ("Read the code"), replace the first sentence with:
`` `CardOCR` crops the bottom-right 55%×14% of the card, enlarges it 3×, and `CardNumberParser` extracts `OP05-119`-style codes, tolerating OCR digit confusions (S→5, O→0, I→1, B→8, …) and a missing dash. The first candidate that exists in the catalog wins.`` Keep the rest of the step (the rotation fallback).

Under the baseline section's comparability note, add one line: `OCR hardening (crop + 3× upscale + tolerant parser): OCR accuracy <before>% → <after>%, top-1 <before>% → <after>% (\`results.csv\` rows \`code-first-featureprint\` and \`ocr-hardened\`).` Fill in the numbers from Steps 1 and 8.

- [ ] **Step 10: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/CardVision/CardOCR.swift \
        apps/ios/Packages/OnePieceKit/Sources/CardVision/CardRecognizer.swift \
        apps/ios/Packages/OnePieceKit/Tests/CardVisionTests/CardVisionTests.swift \
        ml/results/results.csv docs/cv-pipeline.md
git commit -m "Crop and enlarge the number region before OCR; take the first catalog code"
```
`git status` must show nothing else staged. `data/cards/printings.json` may show `embeddingRow` changes if you regenerated embeddings. **Do not stage it.**

---

### Task 3: `ScanRecord` and `ScanLabel` in OnePieceKit

**Files:**
- Create: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/ScanRecord.swift`
- Create: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/ScanRecordTests.swift`
- Delete: `apps/ios/OnePieceAR/Models/ScanRecord.swift`
- Modify: `apps/ios/OnePieceAR/Services/ScanLogger.swift` (`log`, `markCorrected` → `resolve`)
- Modify: `apps/ios/OnePieceAR/App/AppModel.swift` (`correct(to:)`: one call site)

**Interfaces:**
- Consumes: `RecognitionMethod` (OnePieceKit).
- Produces:
  ```swift
  public enum ScanLabel: String, Codable, Sendable {
      case unlabeled = "none", confirmed, corrected
      public init(finalPrintingID: String, firstGuess: String)   // same → confirmed, else corrected
  }
  public struct ScanRecord: Codable, Sendable, Equatable {
      public struct Candidate: Codable, Sendable, Equatable {
          public let printingID: String; public let similarity: Float?; public let matchesOCR: Bool
          public init(printingID: String, similarity: Float?, matchesOCR: Bool)
      }
      public let id: String, date: Date, ocrCardID: String?
      public let method: RecognitionMethod?      // nil in records logged before Phase 2
      public let groupSize: Int                  // 0 when unknown / vision-only
      public let candidates: [Candidate]
      public let spawnedPrintingID: String       // the first guess shown (spawned or info panel)
      public private(set) var finalPrintingID: String
      public private(set) var corrected: Bool
      public private(set) var label: ScanLabel
      public init(id:date:ocrCardID:method:groupSize:candidates:spawnedPrintingID:)
      public mutating func resolve(to printingID: String)
  }
  ```
  App: `ScanLogger.log(_ result: RecognitionResult, spawnedPrintingID: String) throws -> String` (same signature, new fields filled). `ScanLogger.resolve(scanID: String, to printingID: String) throws` replaces `markCorrected`.

- [ ] **Step 1: Write the failing tests**

`Tests/OnePieceKitTests/ScanRecordTests.swift`:
```swift
import Foundation
import Testing
@testable import OnePieceKit

@Suite struct ScanRecordTests {
    static func record() -> ScanRecord {
        ScanRecord(
            id: "20260930-120000_abcd1234", date: Date(timeIntervalSince1970: 1_790_000_000), ocrCardID: "OP05-119",
            method: .ocrVision, groupSize: 3,
            candidates: [.init(printingID: "OP05-119_p1", similarity: 0.91, matchesOCR: true),
                         .init(printingID: "OP05-119", similarity: nil, matchesOCR: true)],
            spawnedPrintingID: "OP05-119_p1")
    }

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    @Test func newRecordIsUnlabeled() {
        let record = Self.record()
        #expect(record.label == .unlabeled && !record.corrected && record.finalPrintingID == "OP05-119_p1")
    }

    @Test func resolveComparesAgainstFirstGuess() {
        var record = Self.record()
        record.resolve(to: "OP05-119_p1")
        #expect(record.label == .confirmed && !record.corrected)
        record.resolve(to: "OP05-119")
        #expect(record.label == .corrected && record.corrected && record.finalPrintingID == "OP05-119")
        record.resolve(to: "OP05-119")   // confirming the correction keeps it a correction
        #expect(record.label == .corrected)
        record.resolve(to: "OP05-119_p1")   // back to the first guess
        #expect(record.label == .confirmed && !record.corrected)
    }

    @Test func roundTripsWithLabelStrings() throws {
        var record = Self.record()
        record.resolve(to: "OP05-119")
        let data = try Self.encoder.encode(record)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains(#""label":"corrected""#) && json.contains(#""method":"ocr+vision""#))
        #expect(json.contains(#""corrected":true"#) && json.contains(#""finalPrintingID":"OP05-119""#))
        #expect(try Self.decoder.decode(ScanRecord.self, from: data) == record)
        #expect(try String(data: Self.encoder.encode(ScanLabel.unlabeled), encoding: .utf8) == #""none""#)
    }

    @Test func decodesPhase1Record() throws {
        // Written by the Phase 1 app: no method, groupSize, or label; a nil similarity is omitted.
        let json = #"""
        {"candidates":[{"matchesOCR":true,"printingID":"OP06-118","similarity":0.84},{"matchesOCR":true,"printingID":"OP06-118_p1"}],
         "corrected":true,"date":"2026-09-29T10:00:00Z","finalPrintingID":"OP06-118_p1","id":"20260929-100000_ffff0000",
         "ocrCardID":"OP06-118","spawnedPrintingID":"OP06-118"}
        """#
        var record = try Self.decoder.decode(ScanRecord.self, from: Data(json.utf8))
        #expect(record.method == nil && record.groupSize == 0 && record.label == .corrected)
        #expect(record.candidates[1].similarity == nil)
        record.resolve(to: "OP06-118")
        #expect(record.label == .confirmed)

        let unlabeled = json.replacingOccurrences(of: #""corrected":true"#, with: #""corrected":false"#)
        #expect(try Self.decoder.decode(ScanRecord.self, from: Data(unlabeled.utf8)).label == .unlabeled)
    }

    @Test func labelFromFinalAndFirstGuess() {
        #expect(ScanLabel(finalPrintingID: "A", firstGuess: "A") == .confirmed)
        #expect(ScanLabel(finalPrintingID: "B", firstGuess: "A") == .corrected)
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter ScanRecordTests`
Expected: build error, `cannot find 'ScanRecord' in scope`.

- [ ] **Step 3: Implement** `Sources/OnePieceKit/Recognition/ScanRecord.swift`

```swift
import Foundation

/// How the user resolved a scan. The training set uses only `confirmed` and `corrected` scans.
public enum ScanLabel: String, Codable, Sendable {
    /// The user never confirmed or corrected the pick.
    case unlabeled = "none"
    /// The final printing is the first guess.
    case confirmed
    /// The final printing differs from the first guess.
    case corrected

    public init(finalPrintingID: String, firstGuess: String) {
        self = finalPrintingID == firstGuess ? .confirmed : .corrected
    }
}

/// One logged scan, written as scan.json next to crop.jpg in the app's Documents/Scans. This is the
/// device-side half of the learning loop; the Python lab reads these files, so existing keys keep
/// their names (`spawnedPrintingID` is the first guess even when it only showed the info panel).
public struct ScanRecord: Codable, Sendable, Equatable {
    public struct Candidate: Codable, Sendable, Equatable {
        public let printingID: String
        /// `nil` when the printing had no embedding score (single-printing code, or no reference row).
        public let similarity: Float?
        public let matchesOCR: Bool

        public init(printingID: String, similarity: Float?, matchesOCR: Bool) {
            self.printingID = printingID
            self.similarity = similarity
            self.matchesOCR = matchesOCR
        }
    }

    public let id: String
    public let date: Date
    public let ocrCardID: String?
    /// `nil` in records logged before recognition methods were recorded.
    public let method: RecognitionMethod?
    /// Printings sharing the read code; 0 for vision-only results and older records.
    public let groupSize: Int
    /// Ranked, best first, as shown in the group list.
    public let candidates: [Candidate]
    /// The first guess (spawned, or shown in the info panel).
    public let spawnedPrintingID: String
    /// What the user ended up with; equals `spawnedPrintingID` unless corrected.
    public private(set) var finalPrintingID: String
    /// Kept for older readers: `label == .corrected`.
    public private(set) var corrected: Bool
    public private(set) var label: ScanLabel

    public init(
        id: String, date: Date, ocrCardID: String?, method: RecognitionMethod?, groupSize: Int,
        candidates: [Candidate], spawnedPrintingID: String
    ) {
        self.id = id
        self.date = date
        self.ocrCardID = ocrCardID
        self.method = method
        self.groupSize = groupSize
        self.candidates = candidates
        self.spawnedPrintingID = spawnedPrintingID
        finalPrintingID = spawnedPrintingID
        corrected = false
        label = .unlabeled
    }

    /// The user confirmed `printingID` (the current pick) or chose it instead.
    public mutating func resolve(to printingID: String) {
        finalPrintingID = printingID
        label = ScanLabel(finalPrintingID: printingID, firstGuess: spawnedPrintingID)
        corrected = label == .corrected
    }

    enum CodingKeys: String, CodingKey {
        case id, date, ocrCardID, method, groupSize, candidates, spawnedPrintingID, finalPrintingID, corrected, label
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        date = try container.decode(Date.self, forKey: .date)
        ocrCardID = try container.decodeIfPresent(String.self, forKey: .ocrCardID)
        method = try container.decodeIfPresent(RecognitionMethod.self, forKey: .method)
        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 0
        candidates = try container.decode([Candidate].self, forKey: .candidates)
        spawnedPrintingID = try container.decode(String.self, forKey: .spawnedPrintingID)
        finalPrintingID = try container.decode(String.self, forKey: .finalPrintingID)
        corrected = try container.decode(Bool.self, forKey: .corrected)
        // Older records have no label: a correction is still a correction, anything else is unlabeled.
        label = try container.decodeIfPresent(ScanLabel.self, forKey: .label) ?? (corrected ? .corrected : .unlabeled)
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$SWIFT_TEST --filter ScanRecordTests`
Expected: PASS.

- [ ] **Step 5: Switch the app to the package type**

- Delete `apps/ios/OnePieceAR/Models/ScanRecord.swift`: `git rm apps/ios/OnePieceAR/Models/ScanRecord.swift`. The Xcode project uses synchronized folders, so no pbxproj edit is needed.
- In `ScanLogger.swift`, replace the `let record = ScanRecord(...)` construction in `log` with:
```swift
        let record = ScanRecord(
            id: id,
            date: .now,
            ocrCardID: result.ocrCardID,
            method: result.method,
            groupSize: result.groupSize,
            candidates: result.candidates.map {
                .init(printingID: $0.printingID, similarity: $0.similarity, matchesOCR: $0.cardID == result.ocrCardID)
            },
            spawnedPrintingID: spawnedPrintingID)
```
  and replace `markCorrected(scanID:finalPrintingID:)` with:
```swift
    /// Records the user's answer for a logged scan: confirming the pick or choosing another printing.
    func resolve(scanID: String, to printingID: String) throws {
        let folder = Self.scansDirectory.appending(path: scanID, directoryHint: .isDirectory)
        let data = try Data(contentsOf: folder.appending(path: "scan.json"))
        var record = try Self.decoder.decode(ScanRecord.self, from: data)
        record.resolve(to: printingID)
        try write(record, to: folder)
    }
```
  Update the class doc comment's first line to: `/// Logs every scan (crop + ranked candidates + the user's confirm/correct label) to`.
- In `AppModel.correct(to:)`, replace `try? await scanLog.markCorrected(scanID: scanID, finalPrintingID: printing.id)` with `try? await scanLog.resolve(scanID: scanID, to: printing.id)`. Task 6 rewrites this function; keep it minimal here.

- [ ] **Step 6: Build and test**

Run: `make test && make build`
Expected: all tests pass and the build succeeds.

- [ ] **Step 7: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/ScanRecord.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/ScanRecordTests.swift \
        apps/ios/OnePieceAR/Services/ScanLogger.swift apps/ios/OnePieceAR/App/AppModel.swift
git commit -m "Label scans confirmed/corrected and record method and group size"
```
(The deletion is already staged by `git rm`.)

---

### Task 4: On-demand thumbnails for catalog printings

**Files:**
- Create: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Catalog/ThumbnailStore.swift`
- Create: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/ThumbnailStoreTests.swift`
- Create: `apps/ios/OnePieceAR/Features/Scanner/CatalogThumbnail.swift`
- Modify: `apps/ios/OnePieceAR/App/AppModel.swift` (store `fullCatalog`, add `thumbnails`, `thumbnail(for:)`)

**Interfaces:**
- Consumes: `CatalogEntry` (`printingId`, `artUrl`).
- Produces:
  - `ThumbnailStore` (actor): `init(directory: URL, fetch: @escaping ThumbnailStore.Fetch = ThumbnailStore.download)`, `func data(for entry: CatalogEntry) async -> Data?`, `typealias Fetch = @Sendable (URL) async throws -> Data`, `static let download: Fetch`.
  - App: `AppModel.fullCatalog: FullCatalog` (`private(set) var`, set in `bootstrap`), `AppModel.thumbnail(for printingID: String) async -> CGImage?`, and the view `CatalogThumbnail(printingID: String)`.

- [ ] **Step 1: Write the failing tests**

`Tests/OnePieceKitTests/ThumbnailStoreTests.swift`:
```swift
import Foundation
import Testing
@testable import OnePieceKit

@Suite struct ThumbnailStoreTests {
    actor Fetcher {
        var calls = 0
        var failing = false
        func setFailing(_ value: Bool) { failing = value }
        func fetch(_ url: URL) throws -> Data {
            calls += 1
            if failing { throw URLError(.notConnectedToInternet) }
            return Data("jpeg:\(url.lastPathComponent)".utf8)
        }
    }

    func directory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "thumbs-\(UUID())", directoryHint: .isDirectory)
    }

    func entry(_ id: String, art: String? = "https://example.com/art.jpg") -> CatalogEntry {
        CatalogEntry(printingId: id, cardId: id, name: "N", set: "OP-01", kind: "base", rarity: "C", artUrl: art)
    }

    @Test func fetchesOnceThenServesFromDisk() async throws {
        let fetcher = Fetcher()
        let dir = directory()
        let store = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        let first = await store.data(for: entry("OP01-001"))
        let second = await store.data(for: entry("OP01-001"))
        #expect(first == Data("jpeg:art.jpg".utf8) && second == first)
        #expect(await fetcher.calls == 1)
        #expect(FileManager.default.fileExists(atPath: dir.appending(path: "OP01-001.jpg").path))
        // A new store over the same directory also hits the cache.
        let reopened = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        #expect(await reopened.data(for: entry("OP01-001")) == first)
        #expect(await fetcher.calls == 1)
    }

    @Test func failedFetchIsNotCachedAndRetries() async {
        let fetcher = Fetcher()
        await fetcher.setFailing(true)
        let dir = directory()
        let store = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        #expect(await store.data(for: entry("OP01-002")) == nil)
        #expect(!FileManager.default.fileExists(atPath: dir.appending(path: "OP01-002.jpg").path))
        await fetcher.setFailing(false)
        #expect(await store.data(for: entry("OP01-002")) != nil)
        #expect(await fetcher.calls == 2)
    }

    @Test func noArtURLNeverFetches() async {
        let fetcher = Fetcher()
        let store = ThumbnailStore(directory: directory()) { try await fetcher.fetch($0) }
        #expect(await store.data(for: entry("OP01-003", art: nil)) == nil)
        #expect(await fetcher.calls == 0)
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter ThumbnailStoreTests`
Expected: build error, `cannot find 'ThumbnailStore' in scope`.

- [ ] **Step 3: Implement** `Sources/OnePieceKit/Catalog/ThumbnailStore.swift`

```swift
import Foundation

/// Card art for catalog printings, fetched on demand from each printing's `artUrl` and cached on
/// disk as `<printingId>.jpg`. Only lists use these; recognition never touches the network.
/// Offline (or on any fetch error) `data(for:)` returns nil and caches nothing, so a later call can
/// still succeed.
public actor ThumbnailStore {
    public typealias Fetch = @Sendable (URL) async throws -> Data

    /// Downloads with URLSession, treating non-2xx responses as failures.
    public static let download: Fetch = { url in
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        return data
    }

    private let directory: URL
    private let fetch: Fetch

    public init(directory: URL, fetch: @escaping Fetch = ThumbnailStore.download) {
        self.directory = directory
        self.fetch = fetch
    }

    public func data(for entry: CatalogEntry) async -> Data? {
        let file = directory.appending(path: "\(entry.printingId).jpg")
        if let cached = try? Data(contentsOf: file) { return cached }
        guard let string = entry.artUrl, let url = URL(string: string),
              let data = try? await fetch(url), !data.isEmpty else { return nil }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: file, options: .atomic)
        return data
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$SWIFT_TEST --filter ThumbnailStoreTests`
Expected: PASS.

- [ ] **Step 5: App wiring**

In `AppModel.swift`:
- Add `import ImageIO` at the top.
- Add a stored property below `catalog`:
```swift
    /// Every printing recognition can identify (catalog.json), or the roster when it isn't bundled.
    private(set) var fullCatalog = FullCatalog(entries: [])
```
- Below `@ObservationIgnored let scanLog = ScanLogger()`, add:
```swift
    @ObservationIgnored let thumbnails = ThumbnailStore(
        directory: URL.cachesDirectory.appending(path: "Thumbnails", directoryHint: .isDirectory))
```
- In `bootstrap`, the local `let fullCatalog: FullCatalog` / `if … else …` block assigns a local. Change it to assign the property: delete the `let fullCatalog: FullCatalog` line, and inside the `if`/`else` write `self.fullCatalog = ...`. Then pass `fullCatalog: self.fullCatalog` to `recognition.prepare`.
- Add at the end of the class (before the closing brace), under `// MARK: Catalog display`:
```swift
    // MARK: Catalog display

    /// Art for any catalog printing: bundled roster art when there is some, otherwise the cached or
    /// downloaded thumbnail. `nil` offline or when the printing has no art URL.
    func thumbnail(for printingID: String) async -> CGImage? {
        if let printing = catalog.printing(id: printingID), let art = assets.cardArt(for: printing) { return art }
        guard let entry = fullCatalog.entry(id: printingID), let data = await thumbnails.data(for: entry),
              let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
```

Create `apps/ios/OnePieceAR/Features/Scanner/CatalogThumbnail.swift`:
```swift
import SwiftUI

/// Card art for any catalog printing, loaded on demand. Shows the placeholder until it loads, and
/// keeps showing it offline.
struct CatalogThumbnail: View {
    @Environment(AppModel.self) private var model
    let printingID: String
    @State private var image: CGImage?

    var body: some View {
        CardArtThumbnail(image: image)
            .task(id: printingID) { image = await model.thumbnail(for: printingID) }
    }
}
```
(`CardArtThumbnail` already exists in `CollectionView.swift`. `CatalogThumbnail` gets its first call sites in Task 6.)

- [ ] **Step 6: Build and test**

Run: `make test && make build`
Expected: both succeed.

- [ ] **Step 7: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Catalog/ThumbnailStore.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/ThumbnailStoreTests.swift \
        apps/ios/OnePieceAR/Features/Scanner/CatalogThumbnail.swift apps/ios/OnePieceAR/App/AppModel.swift
git commit -m "Load catalog thumbnails on demand with a disk cache"
```

---

### Task 5: Detection quad, recognition attempts, and the group summary

**Files:**
- Create: `apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/CardQuad.swift`
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardDetector.swift` (`detect(in:)`, `DetectedCard`)
- Modify: `apps/ios/Packages/OnePieceKit/Sources/CardVision/CardRecognizer.swift` (`RecognitionAttempt`, `attempt(photo:)`, `groupSummary`)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift` (new `CardQuadTests` suite)
- Modify: `apps/ios/Packages/OnePieceKit/Tests/CardVisionTests/CardVisionTests.swift` (new tests)
- Modify: `apps/ios/OnePieceAR/Services/RecognitionService.swift` (`recognize` → `attempt`)
- Modify: `apps/ios/OnePieceAR/App/AppModel.swift` (`consider`: one call)

**Interfaces:**
- Produces (OnePieceKit):
  ```swift
  public struct CardQuad: Equatable, Sendable {
      public let topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint   // Vision-normalized, origin bottom-left, portrait image
      public init(topLeft:topRight:bottomRight:bottomLeft:)
      public var corners: [CGPoint] { get }          // TL, TR, BR, BL
      public var sensorCorners: [CGPoint] { get }    // camera-sensor normalized (landscape, origin top-left)
  }
  ```
- Produces (CardVision):
  ```swift
  public struct DetectedCard: Sendable { public let crop: CGImage; public let quad: CardQuad }
  CardDetector.detect(in: CIImage) throws -> DetectedCard?            // detectCard(in:) stays, returns detect(in:)?.crop
  public struct RecognitionAttempt: Sendable { public let quad: CardQuad?; public let result: RecognitionResult? }
  CardRecognizer.attempt(photo: CIImage) throws -> RecognitionAttempt   // recognize(photo:) stays, returns attempt(photo:).result
  extension RecognitionResult { public var groupSummary: String { get } }
  ```
- Produces (app): `RecognitionService.attempt(_ frame: PixelBufferBox) throws -> RecognitionAttempt?`, which replaces `recognize(_:)`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/OnePieceKitTests/RecognitionTests.swift`:
```swift
@Suite struct CardQuadTests {
    @Test func sensorCornersRotateBackToLandscape() {
        // Full-frame quad in the portrait image (Vision coords, origin bottom-left).
        let quad = CardQuad(topLeft: CGPoint(x: 0, y: 1), topRight: CGPoint(x: 1, y: 1),
                            bottomRight: CGPoint(x: 1, y: 0), bottomLeft: CGPoint(x: 0, y: 0))
        // The portrait image is the sensor image rotated 90° clockwise (`.oriented(.right)`):
        // portrait top-left came from sensor bottom-left, top-right from top-left, and so on.
        #expect(quad.sensorCorners == [CGPoint(x: 0, y: 1), CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1)])
        let point = CardQuad(topLeft: CGPoint(x: 0.25, y: 0.75), topRight: .zero, bottomRight: .zero, bottomLeft: .zero)
        #expect(point.sensorCorners[0] == CGPoint(x: 0.25, y: 0.75))
    }
}
```
(Check of the second case: `x' = 1 - y = 0.25`, `y' = 1 - x = 0.75`.)

Append to `Tests/CardVisionTests/CardVisionTests.swift`, inside `CardVisionTests`:
```swift
    @Test func detectReportsTheCardQuad() throws {
        let detected = try #require(try CardDetector(context: context).detect(in: photo(of: syntheticCard(seed: 5))))
        // photo(of:) places the card at x 290/900…605/900 and y 380/1200…820/1200 (origin bottom-left).
        let quad = detected.quad
        #expect(abs(quad.topLeft.x - 290.0 / 900) < 0.02 && abs(quad.topLeft.y - 820.0 / 1200) < 0.02)
        #expect(abs(quad.bottomRight.x - 605.0 / 900) < 0.02 && abs(quad.bottomRight.y - 380.0 / 1200) < 0.02)
        #expect(detected.crop.width == 630)
    }

    @Test func attemptWithoutACardHasNoQuad() throws {
        let engine = EmbeddingEngine()
        let ids = ["OP01-010"]
        let embeddings = try PrintingEmbeddings.build(from: [
            (printingID: ids[0], vector: try CardRecognizer.referenceEmbedding(for: CIImage(cgImage: syntheticCard(seed: 10)), engine: engine, context: context)),
        ])
        let recognizer = CardRecognizer(engine: engine, matcher: VariantMatcher(catalog: FullCatalog(printingIDs: ids), embeddings: embeddings, source: .bundledIndex), context: context)
        let empty = CIImage(color: CIColor(red: 0.08, green: 0.08, blue: 0.1)).cropped(to: CGRect(x: 0, y: 0, width: 900, height: 1200))
        let attempt = try recognizer.attempt(photo: empty)
        #expect(attempt.quad == nil && attempt.result == nil)
        let found = try recognizer.attempt(photo: photo(of: syntheticCard(seed: 10)))
        #expect(found.quad != nil)
    }
```
And inside `CodeFirstRecognitionTests`:
```swift
    @Test func groupSummaryText() throws {
        let crop = syntheticCard(seed: 95)
        let candidate = RecognitionCandidate(printingID: "OP05-119", cardID: "OP05-119", similarity: nil)
        func result(_ method: RecognitionMethod, group: Int, count: Int) -> RecognitionResult {
            RecognitionResult(crop: crop, candidates: Array(repeating: candidate, count: count), ocrCardID: "OP05-119",
                              flipped: false, method: method, groupSize: group)
        }
        #expect(result(.ocrVision, group: 4, count: 4).groupSummary == "OP05-119 · 4 printings")
        #expect(result(.ocrUnique, group: 1, count: 1).groupSummary == "OP05-119 · 1 printing")
        #expect(result(.visionOnly, group: 0, count: 5).groupSummary == "Matched by art · 5 candidates")
        #expect(result(.visionOnly, group: 0, count: 1).groupSummary == "Matched by art · 1 candidate")
    }
```
(`RecognitionResult`'s memberwise init is internal, which `@testable import CardVision` allows.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$SWIFT_TEST --filter "CardQuadTests|CardVisionTests|CodeFirstRecognitionTests"`
Expected: build errors (`CardQuad`, `detect(in:)`, `attempt(photo:)`, `groupSummary` not found).

- [ ] **Step 3: Implement `CardQuad`** (`Sources/OnePieceKit/Recognition/CardQuad.swift`)

```swift
import CoreGraphics

/// Where a card was detected: its corners in Vision's normalized coordinates (origin bottom-left)
/// of the portrait image recognition ran on.
public struct CardQuad: Equatable, Sendable {
    public let topLeft: CGPoint
    public let topRight: CGPoint
    public let bottomRight: CGPoint
    public let bottomLeft: CGPoint

    public init(topLeft: CGPoint, topRight: CGPoint, bottomRight: CGPoint, bottomLeft: CGPoint) {
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomRight = bottomRight
        self.bottomLeft = bottomLeft
    }

    /// Top-left, top-right, bottom-right, bottom-left.
    public var corners: [CGPoint] { [topLeft, topRight, bottomRight, bottomLeft] }

    /// The corners in the camera sensor's normalized image coordinates (landscape, origin top-left),
    /// which `ARFrame.displayTransform(for:viewportSize:)` maps onto the screen. The portrait image
    /// is the sensor image rotated 90° clockwise (`CIImage.oriented(.right)`), so a portrait point
    /// (x, y) with origin bottom-left came from sensor point (1 - y, 1 - x).
    public var sensorCorners: [CGPoint] {
        corners.map { CGPoint(x: 1 - $0.y, y: 1 - $0.x) }
    }
}
```

- [ ] **Step 4: Implement detection with the quad** (`CardDetector.swift`)

Add above `public struct CardDetector`:
```swift
/// A detected card: the rectified canonical crop and where it was in the image.
public struct DetectedCard: Sendable {
    public let crop: CGImage
    public let quad: CardQuad
}
```
Replace `detectCard(in:)` with:
```swift
    /// - Parameter image: image already rotated to match what the user sees (portrait).
    public func detect(in image: CIImage) throws -> DetectedCard? {
        let request = VNDetectRectanglesRequest()
        // Wider than the card's 0.716 on the high side: a card tilted away from the camera
        // foreshortens and looks squarer (eval: 29% detection on angled shots with ±0.08).
        request.minimumAspectRatio = Self.aspectRange.lowerBound
        request.maximumAspectRatio = Self.aspectRange.upperBound
        request.minimumSize = 0.2
        request.minimumConfidence = 0.5
        request.quadratureTolerance = 30
        request.maximumObservations = 1

        try VNImageRequestHandler(ciImage: image, options: [:]).perform([request])
        guard let rectangle = request.results?.first, let crop = rectify(image, rectangle) else { return nil }
        let quad = CardQuad(topLeft: rectangle.topLeft, topRight: rectangle.topRight,
                            bottomRight: rectangle.bottomRight, bottomLeft: rectangle.bottomLeft)
        return DetectedCard(crop: crop, quad: quad)
    }

    /// The rectified card only.
    public func detectCard(in image: CIImage) throws -> CGImage? {
        try detect(in: image)?.crop
    }
```

- [ ] **Step 5: Implement attempts and the summary** (`CardRecognizer.swift`)

Add after `struct RecognitionResult`:
```swift
extension RecognitionResult {
    /// "OP05-119 · 4 printings", or "Matched by art · 5 candidates" when no catalog code was read.
    public var groupSummary: String {
        switch method {
        case .ocrUnique, .ocrVision:
            "\(ocrCardID ?? "?") · \(groupSize) printing\(groupSize == 1 ? "" : "s")"
        case .visionOnly:
            "Matched by art · \(candidates.count) candidate\(candidates.count == 1 ? "" : "s")"
        }
    }
}

/// One camera frame's outcome, for the live debug overlay: where the card was (if anywhere) and the
/// recognition result (if one passed).
public struct RecognitionAttempt: Sendable {
    /// `nil` when no card-shaped rectangle was detected.
    public let quad: CardQuad?
    /// `nil` when nothing was detected or a vision-only match fell below the threshold.
    public let result: RecognitionResult?
}
```
Replace `recognize(photo:)` with:
```swift
    /// Photo or camera frame: find the card, then recognize it. Reports the detected quad even when
    /// recognition returns nothing.
    public func attempt(photo: CIImage) throws -> RecognitionAttempt {
        guard let card = try detector.detect(in: photo) else { return RecognitionAttempt(quad: nil, result: nil) }
        return RecognitionAttempt(quad: card.quad, result: try recognize(canonicalCard: card.crop))
    }

    /// Photo or camera frame: find the card first. `nil` when no card-shaped rectangle is found.
    public func recognize(photo: CIImage) throws -> RecognitionResult? {
        try attempt(photo: photo).result
    }
```

- [ ] **Step 6: App call sites**

In `RecognitionService.swift`, replace `recognize(_:)` with:
```swift
    /// - Parameter frame: ARKit's captured image (landscape sensor orientation).
    func attempt(_ frame: PixelBufferBox) throws -> RecognitionAttempt? {
        // Portrait UI: the sensor image must be rotated to match what the user sees.
        try recognizer?.attempt(photo: CIImage(cvPixelBuffer: frame.buffer).oriented(.right))
    }
```
In `AppModel.consider(_:)`, replace `let result = try? await recognition.recognize(buffer)` with:
```swift
            let result = (try? await recognition.attempt(buffer))?.result
```
(Task 6 rewrites `consider`. This only keeps the app compiling.)

- [ ] **Step 7: Test and build**

Run: `make test && make build`
Expected: both succeed.

- [ ] **Step 8: Commit**

```bash
git add apps/ios/Packages/OnePieceKit/Sources/OnePieceKit/Recognition/CardQuad.swift \
        apps/ios/Packages/OnePieceKit/Sources/CardVision/CardDetector.swift \
        apps/ios/Packages/OnePieceKit/Sources/CardVision/CardRecognizer.swift \
        apps/ios/Packages/OnePieceKit/Tests/OnePieceKitTests/RecognitionTests.swift \
        apps/ios/Packages/OnePieceKit/Tests/CardVisionTests/CardVisionTests.swift \
        apps/ios/OnePieceAR/Services/RecognitionService.swift apps/ios/OnePieceAR/App/AppModel.swift
git commit -m "Report the detected card quad and a group summary for each attempt"
```

---

### Task 6: Scan flow UI: strip, group sheet, info panel, confirm/correct, debug overlay

The app target has no unit-test target. Verification is `make build` plus the device checklist in Step 9. Keep all decision logic in `AppModel` methods named below, so the views stay declarative.

**Files:**
- Modify: `apps/ios/OnePieceAR/App/AppModel.swift` (scan state, `select`, `consider`, new methods; remove `correct(to:)` and `lastRecognition`)
- Modify: `apps/ios/OnePieceAR/AR/ARSessionManager.swift` (`displayTransform(viewportSize:)`)
- Modify: `apps/ios/OnePieceAR/Features/Scanner/ExperienceView.swift` (overlay, sheets, bottom controls; move the debug view out)
- Create: `apps/ios/OnePieceAR/Features/Scanner/ScanResultStrip.swift`
- Create: `apps/ios/OnePieceAR/Features/Scanner/GroupSheet.swift`
- Create: `apps/ios/OnePieceAR/Features/Scanner/CardInfoPanel.swift`
- Create: `apps/ios/OnePieceAR/Features/Scanner/RecognitionDebugOverlay.swift`
- Delete: `apps/ios/OnePieceAR/Features/Scanner/AlternativesSheet.swift`
- Modify: `apps/ios/OnePieceAR/Features/Settings/SettingsView.swift` (footer text)
- Modify: `docs/cv-pipeline.md` (step 6 and the "Scan logs" section)

**Interfaces:**
- Consumes:
  - `ScanLabel`, and `ScanLogger.log(_:spawnedPrintingID:)` / `resolve(scanID:to:)` (Task 3)
  - `fullCatalog`, `thumbnail(for:)`, `CatalogThumbnail` (Task 4)
  - `RecognitionAttempt`, `CardQuad.sensorCorners`, `RecognitionResult.groupSummary`, `RecognitionService.attempt` (Task 5)
- Produces (AppModel):
  ```swift
  struct ScanOutcome { let slot: Int; let result: RecognitionResult; let scanID: String?; var pickID: String; var label: ScanLabel }
  private(set) var lastScan: ScanOutcome?
  private(set) var identified: CatalogEntry?
  private(set) var debugAttempt: RecognitionAttempt?
  var showingGroup: Bool          // replaces showingAlternatives
  var correctingScan: Bool
  func choose(_ printingID: String) async
  func confirmPick() async
  func isInRoster(_ printingID: String) -> Bool
  func displayName(for printingID: String) -> String
  func select(_ printing: Printing, slot: Int? = nil, crop: CGImage? = nil, scanID: String? = nil, fromScan: Bool = false) async
  ```
  `ARSessionManager.displayTransform(viewportSize: CGSize) -> CGAffineTransform?`

- [ ] **Step 1: `AppModel` scan state**

Replace the `lastRecognition` property and its doc comment with:
```swift
    /// The most recent scan: what recognition returned, what's shown now, and how the user labeled it.
    struct ScanOutcome {
        let slot: Int
        let result: RecognitionResult
        let scanID: String?
        /// The printing shown now: spawned if it's in the roster, otherwise in the info panel.
        var pickID: String
        var label: ScanLabel
    }

    private(set) var lastScan: ScanOutcome?
    /// A recognized printing outside the roster, shown in the card-info panel instead of spawning.
    private(set) var identified: CatalogEntry?
    /// The latest frame's outcome while scanning with the debug overlay on (including no-match frames).
    private(set) var debugAttempt: RecognitionAttempt?
```
Replace `var showingAlternatives = false` with:
```swift
    var showingGroup = false
    /// Set when "None of these" opens the picker, so the pick is recorded as the scan's correction.
    var correctingScan = false
```
In `clearAll()`, replace `lastRecognition = nil` with `lastScan = nil` and add `identified = nil`.

- [ ] **Step 2: `select` and the scan loop**

Change `select`'s signature and doc comment to:
```swift
    /// Spawns the variant for a printing in a slot and starts anchoring it.
    /// - Parameters:
    ///   - crop: the scanned card image, used as the tracking image when no card art is bundled.
    ///   - scanID: set when this selection came from a logged scan.
    ///   - fromScan: the selection shows the last scan's pick, so `lastScan` stays.
    func select(_ printing: Printing, slot requestedSlot: Int? = nil, crop: CGImage? = nil, scanID: String? = nil, fromScan: Bool = false) async {
```
and replace its line `if scanID == nil, lastRecognition?.slot == slot { lastRecognition = nil }` with:
```swift
        identified = nil
        if !fromScan, lastScan?.slot == slot { lastScan = nil }
```
In `startScan()`, after the `guard`, add `identified = nil`. In `stopScan()`, add `debugAttempt = nil`.

Replace the whole `consider(_:)` and `correct(to:)` with:
```swift
    private func consider(_ frame: ARFrame) {
        let now = Date.now
        guard !recognitionBusy, now.timeIntervalSince(lastRecognitionAttempt) > Self.recognitionInterval else { return }
        recognitionBusy = true
        lastRecognitionAttempt = now
        let buffer = PixelBufferBox(buffer: frame.capturedImage)
        Task {
            let attempt = try? await recognition.attempt(buffer)   // try? flattens the optional
            recognitionBusy = false
            guard scanState == .searching else { return }
            if settings.showDebug { debugAttempt = attempt }
            guard let result = attempt?.result, let best = result.best else { return }
            stopScan()
            let slot = targetSlot
            let scanID = settings.logScans ? try? await scanLog.log(result, spawnedPrintingID: best.printingID) : nil
            lastScan = ScanOutcome(slot: slot, result: result, scanID: scanID, pickID: best.printingID, label: .unlabeled)
            await show(best.printingID, slot: slot, crop: result.crop, scanID: scanID)
        }
    }

    /// Spawns a roster printing, or shows any other catalog printing in the info panel.
    private func show(_ printingID: String, slot: Int, crop: CGImage?, scanID: String?) async {
        if let printing = catalog.printing(id: printingID) {
            await select(printing, slot: slot, crop: crop, scanID: scanID, fromScan: true)
        } else {
            clear(slot: slot)
            identified = fullCatalog.entry(id: printingID)
        }
    }

    /// The user's answer for the last scan: the pick (a confirmation) or another printing from the
    /// group list or the manual picker (a correction). Labels the scan log and shows the choice.
    func choose(_ printingID: String) async {
        guard var scan = lastScan else { return }
        let changed = printingID != scan.pickID
        scan.pickID = printingID
        scan.label = ScanLabel(finalPrintingID: printingID, firstGuess: scan.result.best?.printingID ?? printingID)
        lastScan = scan
        if let scanID = scan.scanID {
            try? await scanLog.resolve(scanID: scanID, to: printingID)
        }
        if changed {
            await show(printingID, slot: scan.slot, crop: scan.result.crop, scanID: scan.scanID)
        }
    }

    /// The current pick is right.
    func confirmPick() async {
        guard let pick = lastScan?.pickID else { return }
        await choose(pick)
    }
```
Add to the `// MARK: Catalog display` section (from Task 4):
```swift
    func isInRoster(_ printingID: String) -> Bool {
        catalog.printing(id: printingID) != nil
    }

    /// "Monkey.D.Luffy · parallel" for any catalog printing.
    func displayName(for printingID: String) -> String {
        guard let entry = fullCatalog.entry(id: printingID) else { return printingID }
        return "\(entry.name) · \(entry.kind)"
    }
```
`clear(slot:)` is `private` and already used by `show`. It stays private, because both are in `AppModel`.

- [ ] **Step 3: `ARSessionManager.displayTransform`**

Add under `// MARK: Hit testing`:
```swift
    /// Maps normalized camera-sensor coordinates (landscape, origin top-left) to normalized view
    /// coordinates for the portrait UI. `nil` before the first frame.
    func displayTransform(viewportSize: CGSize) -> CGAffineTransform? {
        arView.session.currentFrame?.displayTransform(for: .portrait, viewportSize: viewportSize)
    }
```

- [ ] **Step 4: Debug overlay** (create `Features/Scanner/RecognitionDebugOverlay.swift`, and move the debug text view out of `ExperienceView.swift`)

Delete `private struct RecognitionDebugView` from `ExperienceView.swift`. Create:
```swift
import CardVision
import OnePieceKit
import SwiftUI

/// The detected card's outline, drawn over the camera image for the frame recognition last looked at.
/// Settings > Show recognition debug.
struct RecognitionDebugOverlay: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        GeometryReader { geometry in
            if let quad = model.debugAttempt?.quad,
               let transform = model.session.displayTransform(viewportSize: geometry.size) {
                let points = quad.sensorCorners.map { corner in
                    let normalized = corner.applying(transform)
                    return CGPoint(x: normalized.x * geometry.size.width, y: normalized.y * geometry.size.height)
                }
                Path { path in
                    path.addLines(points)
                    path.closeSubpath()
                }
                .stroke(model.debugAttempt?.result == nil ? Color.yellow : Color.green, lineWidth: 3)
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

/// What recognition decided: OCR read, method, group size, and the top candidates' similarities.
struct RecognitionDebugView: View {
    let attempt: RecognitionAttempt?
    let lastResult: RecognitionResult?

    var body: some View {
        let result = attempt?.result ?? lastResult
        HStack(alignment: .top, spacing: 10) {
            if let result {
                Image(decorative: result.crop, scale: 1)
                    .resizable()
                    .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
                    .frame(width: 60)
            }
            VStack(alignment: .leading, spacing: 2) {
                if let attempt, attempt.result == nil {
                    Text(attempt.quad == nil ? "No card detected" : "Card found, no confident match")
                }
                if let result {
                    Text("OCR: \(result.ocrCardID ?? "–")  \(result.method.rawValue)  n=\(result.groupSize)")
                    ForEach(result.candidates.prefix(3)) { candidate in
                        Text("\(candidate.printingID)  \(candidate.similarity.map { String(format: "%.3f", $0) } ?? "–")")
                    }
                }
            }
            .font(.caption.monospaced())
            Spacer()
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}
```

- [ ] **Step 5: Strip, group sheet, info panel**

Create `Features/Scanner/ScanResultStrip.swift`:
```swift
import CardVision
import OnePieceKit
import SwiftUI

/// After a scan: the pick, its group ("OP05-119 · 4 printings"), a confirm button, and the way into
/// the group list.
struct ScanResultStrip: View {
    @Environment(AppModel.self) private var model
    let scan: AppModel.ScanOutcome

    var body: some View {
        HStack(spacing: 10) {
            CatalogThumbnail(printingID: scan.pickID)
                .frame(width: 34)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName(for: scan.pickID))
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(scan.result.groupSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            if scan.label == .unlabeled {
                Button {
                    Task { await model.confirmPick() }
                } label: {
                    Image(systemName: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .accessibilityLabel("Confirm this printing")
            } else {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                    .accessibilityLabel(scan.label == .confirmed ? "Confirmed" : "Corrected")
            }
            Button {
                model.showingGroup = true
            } label: {
                Image(systemName: "chevron.up")
            }
            .buttonStyle(.bordered)
            .accessibilityLabel("Show all printings")
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .contentShape(Rectangle())
        .onTapGesture { model.showingGroup = true }
    }
}
```
Delete `Features/Scanner/AlternativesSheet.swift` (`git rm`). Create `Features/Scanner/GroupSheet.swift`:
```swift
import CardVision
import OnePieceKit
import SwiftUI

/// Every candidate for the last scan: the printings sharing the read code, or the closest art when
/// no code was read. Tapping a row switches to it; Confirm records that the current pick is right.
struct GroupSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let scan = model.lastScan {
                    Section {
                        HStack(spacing: 12) {
                            Image(decorative: scan.result.crop, scale: 1)
                                .resizable()
                                .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
                                .frame(height: 110)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            VStack(alignment: .leading, spacing: 4) {
                                Text("What the camera saw").font(.headline)
                                Text(scan.result.groupSummary).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    Section(scan.result.method == .visionOnly ? "Closest art" : "Printings of \(scan.result.ocrCardID ?? "")") {
                        ForEach(scan.result.candidates) { candidate in
                            GroupRow(candidate: candidate, isPick: candidate.printingID == scan.pickID) {
                                dismiss()
                                Task { await model.choose(candidate.printingID) }
                            }
                        }
                    }
                }
                Section {
                    Button("None of these, pick manually") {
                        model.correctingScan = true
                        dismiss()
                        model.showingPicker = true
                    }
                }
            }
            .navigationTitle("Which printing?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Confirm") {
                        dismiss()
                        Task { await model.confirmPick() }
                    }
                    .disabled(model.lastScan == nil)
                }
            }
        }
    }
}

private struct GroupRow: View {
    @Environment(AppModel.self) private var model
    let candidate: RecognitionCandidate
    let isPick: Bool
    let onChoose: () -> Void

    var body: some View {
        let entry = model.fullCatalog.entry(id: candidate.printingID)
        Button(action: onChoose) {
            HStack(spacing: 12) {
                CatalogThumbnail(printingID: candidate.printingID)
                    .frame(width: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry?.name ?? candidate.cardID).font(.body.weight(.medium))
                    Text([candidate.printingID, entry?.kind ?? "", entry?.rarity ?? ""].filter { !$0.isEmpty }.joined(separator: " · "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if model.isInRoster(candidate.printingID) {
                        Text("Spawns a character").font(.caption2).foregroundStyle(.tint)
                    }
                }
                Spacer()
                if let similarity = candidate.similarity {
                    Text(similarity, format: .percent.precision(.fractionLength(0)))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if isPick {
                    Image(systemName: "checkmark").foregroundStyle(.tint).accessibilityLabel("Current pick")
                }
            }
        }
        .tint(.primary)
    }
}
```
Create `Features/Scanner/CardInfoPanel.swift`:
```swift
import OnePieceKit
import SwiftUI

/// A recognized card that isn't in the roster: what it is, without a character to spawn.
struct CardInfoPanel: View {
    let entry: CatalogEntry

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            CatalogThumbnail(printingID: entry.printingId)
                .frame(width: 70)
            VStack(alignment: .leading, spacing: 4) {
                Text(entry.name).font(.headline)
                Text("\(entry.cardId) · \(entry.set)").font(.subheadline)
                Text([entry.kind.capitalized, entry.rarity].filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("No character for this card yet")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}
```

- [ ] **Step 6: Wire `ExperienceView`**

- In the `ZStack`, directly after the `if ARSessionManager.isSupported { … } else { … }` block, add:
```swift
            if model.settings.showDebug, model.scanState == .searching {
                RecognitionDebugOverlay()
            }
```
- Replace the `if model.settings.showDebug, let recognition = model.lastRecognition { … }` block with:
```swift
                if model.settings.showDebug, model.debugAttempt != nil || model.lastScan != nil {
                    RecognitionDebugView(attempt: model.debugAttempt, lastResult: model.lastScan?.result)
                }
```
- Replace the picker sheet with:
```swift
        .sheet(isPresented: $model.showingPicker, onDismiss: { model.correctingScan = false }) {
            CollectionView { printing in
                let correcting = model.correctingScan
                model.correctingScan = false
                model.showingPicker = false
                Task {
                    if correcting {
                        await model.choose(printing.id)
                    } else {
                        await model.select(printing)
                    }
                }
            }
        }
```
- Replace the alternatives sheet with:
```swift
        .sheet(isPresented: $model.showingGroup) {
            GroupSheet()
                .presentationDetents([.medium, .large])
        }
```
- In `bottomControls`, replace the `if model.lastRecognition != nil, model.scanState == .off { Button("Not this one?") … }` block with:
```swift
        if model.scanState == .off {
            if let entry = model.identified {
                CardInfoPanel(entry: entry)
            }
            if let scan = model.lastScan {
                ScanResultStrip(scan: scan)
            }
        }
```

- [ ] **Step 7: Settings copy and docs**

In `SettingsView.swift`, change the Recognition section footer text to:
`"Scan logs live in Documents/Scans. Confirm or correct a scan (the ✓ on the result strip, or the printing list) to label it for training. Copy the folder to the Mac from Finder (device > Files > OnePieceAR)."`

In `docs/cv-pipeline.md`:
- Replace step 6 ("Spawn or show") with:
```markdown
6. **Spawn or show.** A roster printing spawns its character right away; any other printing shows a
   card-info panel (name, code, set, kind, rarity). Either way a strip reads "OP05-119 · 4 printings"
   (or "Matched by art · 5 candidates") with the pick; ✓ confirms it, and tapping the strip opens the
   group list, where any row can be chosen instead ("None of these" opens the manual picker).
   Thumbnails there load on demand from the printing's art URL and are cached; offline, rows are text only.
```
- In the "Scan logs (learning loop input)" section, replace the sentence starting "The JSON holds" with:
`The JSON holds the ranked candidates with similarities, the OCR read, \`method\` and \`groupSize\`, the first guess (\`spawnedPrintingID\`), the final printing, and \`label\`: \`confirmed\` (✓ or re-choosing the first guess), \`corrected\` (another printing chosen), or \`none\` (no answer). \`corrected\` is kept as a boolean for older readers. Only labeled scans are meant for training and testing.`
- Add a sentence after the debug-related text (or at the end of step 7) describing the overlay: `Settings > Show recognition debug draws the detected card outline live while scanning (yellow: found, no match; green: recognized) and lists the OCR read, method, and top similarities.`

- [ ] **Step 8: Build and test**

Run: `make build && make test`
Expected: both succeed. Fix any compile errors in the files this task touches. `grep -rn "lastRecognition\|showingAlternatives\|AlternativesSheet\|correct(to:" apps/ios/OnePieceAR` must print nothing.

- [ ] **Step 9: Device checklist** (write the results into the task report; mark any item you couldn't run as NOT RUN)

If an iPhone is available, run the app and check each item. Otherwise list them all as NOT RUN, for the user to do.
1. Scan a roster card (OP05-119 or OP06-118): the character spawns, and the strip shows its name and "OP05-119 · N printings".
2. Tap ✓: it turns into a green seal. `Documents/Scans/<id>/scan.json` has `"label" : "confirmed"`.
3. Open the strip and choose another printing of the same code: the character switches (or the info panel appears if that printing isn't in the roster). The scan's label becomes `corrected`.
4. Scan a non-roster card: no spawn, the info panel shows its name, code, set, kind, and rarity. The strip still works.
5. Airplane mode: open the group list. Rows show text and placeholders, and scanning still recognizes cards.
6. Settings > Show recognition debug, then scan: a yellow or green outline tracks the card, and the debug text updates.
7. "None of these, pick manually" → pick a card: it spawns, and the scan's label is `corrected`.

- [ ] **Step 10: Commit**

```bash
git add apps/ios/OnePieceAR/App/AppModel.swift apps/ios/OnePieceAR/AR/ARSessionManager.swift \
        apps/ios/OnePieceAR/Features/Scanner/ExperienceView.swift \
        apps/ios/OnePieceAR/Features/Scanner/ScanResultStrip.swift apps/ios/OnePieceAR/Features/Scanner/GroupSheet.swift \
        apps/ios/OnePieceAR/Features/Scanner/CardInfoPanel.swift apps/ios/OnePieceAR/Features/Scanner/RecognitionDebugOverlay.swift \
        apps/ios/OnePieceAR/Features/Settings/SettingsView.swift docs/cv-pipeline.md
git rm apps/ios/OnePieceAR/Features/Scanner/AlternativesSheet.swift
git commit -m "Show scan results with confirm/correct, a card-info panel, and a live debug overlay"
```
