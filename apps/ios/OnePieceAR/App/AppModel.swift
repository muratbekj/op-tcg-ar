import ARKit
import CardVision
import ImageIO
import OnePieceKit
import RealityKit

/// App state and orchestration: catalog, card slots, spawning, placement, scanning, correction.
@Observable
final class AppModel {
    enum Mode: String, CaseIterable, Identifiable {
        case solo, battle
        var id: Self { self }
    }

    enum Placement: Equatable {
        /// Tracking the card's reference image; a surface tap also works.
        case waitingForCard
        /// No usable card image; waiting for a surface tap.
        case waitingForSurface
        case placed
    }

    struct Slot {
        let index: Int
        let printing: Printing
        let card: Card
        let variant: CharacterVariant
        var placement: Placement
        let usesPlaceholder: Bool
        let scanID: String?
    }

    enum ScanState: Equatable {
        case off, searching
    }

    private(set) var catalog = CardCatalog(cards: [], printings: [], variants: [])
    /// Every printing recognition can identify (catalog.json), or the roster when it isn't bundled.
    private(set) var fullCatalog = FullCatalog(entries: [])
    private(set) var loadError: String?
    private(set) var mode: Mode = .solo
    private(set) var slots: [Int: Slot] = [:]
    /// Which slot the next scan or pick fills.
    private(set) var targetSlot = 0
    private(set) var scanState: ScanState = .off
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
    private(set) var recognitionSummary: String?
    private(set) var banner: String?

    var showingPicker = false
    var showingGroup = false
    /// Set when "None of these" opens the picker, so the pick is recorded as the scan's correction.
    var correctingScan = false
    var showingSettings = false

    let settings = AppSettings()
    let session = ARSessionManager()
    let battle = BattleCoordinator()
    @ObservationIgnored let assets = AssetService()
    @ObservationIgnored let scanLog = ScanLogger()
    @ObservationIgnored let thumbnails = ThumbnailStore(
        directory: URL.cachesDirectory.appending(path: "Thumbnails", directoryHint: .isDirectory))
    @ObservationIgnored private let recognition = RecognitionService()
    @ObservationIgnored private let spawner: CharacterSpawner
    @ObservationIgnored private var anchors: [Int: CardAnchor] = [:]
    @ObservationIgnored private var controllers: [Int: CharacterController] = [:]
    @ObservationIgnored private var selectionGeneration: [Int: Int] = [:]
    @ObservationIgnored private var recognitionBusy = false
    @ObservationIgnored private var lastRecognitionAttempt = Date.distantPast
    @ObservationIgnored private var bootstrapped = false

    static let recognitionInterval: TimeInterval = 0.35

    init() {
        spawner = CharacterSpawner(assets: assets, vfx: VFXLibrary())
    }

    var slotCount: Int { mode == .solo ? 1 : 2 }
    var allSlotsPlaced: Bool { (0..<slotCount).allSatisfy { slots[$0]?.placement == .placed } }
    var soloSlot: Slot? { slots[0] }
    var canScan: Bool { recognitionSummary != nil }

    /// One line for the status pill, most urgent first.
    var statusText: String? {
        if let message = session.trackingMessage { return message }
        if scanState == .searching { return "Hold the card flat inside the frame" }
        if let pending = (0..<slotCount).compactMap({ slots[$0] }).first(where: { $0.placement != .placed }) {
            return switch pending.placement {
            case .waitingForCard: "Point at \(pending.card.id), or tap a surface to place \(pending.variant.name)"
            case .waitingForSurface: "Tap a surface to place \(pending.card.name) (\(pending.variant.name))"
            case .placed: nil
            }
        }
        return banner
    }

    // MARK: Lifecycle

    func bootstrap() async {
        guard !bootstrapped else { return }
        bootstrapped = true
        do {
            guard let resources = Bundle.main.resourceURL else { throw CocoaError(.fileNoSuchFile) }
            catalog = try CardCatalog.load(from: resources)
            let issues = catalog.validate()
            if !issues.isEmpty { print("Catalog issues:\n" + issues.joined(separator: "\n")) }
        } catch {
            loadError = "Could not load card data: \(error)"
            return
        }

        session.onImageAnchor = { [weak self] anchor in self?.handleImageAnchor(anchor) }
        session.start()

        let art = catalog.printings.compactMap { printing in
            assets.cardArt(for: printing).map { (printingID: printing.id, image: $0) }
        }
        if let url = Bundle.main.url(forResource: "catalog", withExtension: "json"), let loaded = try? FullCatalog.load(from: url) {
            self.fullCatalog = loaded
        } else {
            print("AppModel: no catalog.json bundled; recognizing roster printings only")
            self.fullCatalog = FullCatalog(roster: catalog)
        }
        let bundle = Bundle.main
        if await recognition.prepare(
            catalog: catalog,
            fullCatalog: self.fullCatalog,
            bundledIndex: bundle.url(forResource: "printings", withExtension: "f32"),
            bundledMetadata: bundle.url(forResource: "printings.meta", withExtension: "json"),
            bundledModel: bundle.url(forResource: RecognitionService.modelName, withExtension: "mlmodelc"),
            cardArt: art) {
            recognitionSummary = await recognition.referenceSummary
        }
    }

    func setMode(_ newMode: Mode) {
        guard newMode != mode else { return }
        clearAll()
        mode = newMode
    }

    /// Removes every character and anchor, e.g. to start a new battle.
    func clearAll() {
        stopScan()
        for slot in Array(slots.keys) { clear(slot: slot) }
        battle.reset()
        lastScan = nil
        identified = nil
        targetSlot = 0
        banner = nil
    }

    func applySettings() {
        for controller in controllers.values { controller.wanderEnabled = settings.wanderEnabled }
    }

    // MARK: Summoning

    /// Spawns the variant for a printing in a slot and starts anchoring it.
    /// - Parameters:
    ///   - crop: the scanned card image, used as the tracking image when no card art is bundled.
    ///   - scanID: set when this selection came from a logged scan.
    ///   - fromScan: the selection shows the last scan's pick, so `lastScan` stays.
    func select(_ printing: Printing, slot requestedSlot: Int? = nil, crop: CGImage? = nil, scanID: String? = nil, fromScan: Bool = false) async {
        let slot = requestedSlot ?? targetSlot
        guard let card = catalog.card(for: printing), let variant = catalog.variant(for: printing) else {
            banner = "No character variant for \(printing.id)"
            return
        }
        let generation = selectionGeneration[slot, default: 0] + 1
        selectionGeneration[slot] = generation

        clear(slot: slot)
        identified = nil
        if !fromScan, lastScan?.slot == slot { lastScan = nil }
        slots[slot] = Slot(
            index: slot, printing: printing, card: card, variant: variant, placement: .waitingForCard,
            usesPlaceholder: !assets.hasModel(for: variant), scanID: scanID)
        if mode == .battle, slot == targetSlot { targetSlot = min(slot + 1, slotCount - 1) }

        let anchor = anchor(for: slot)
        let controller = await spawner.spawn(variant, on: anchor)
        guard selectionGeneration[slot] == generation else {
            spawner.despawn(controller)
            return
        }
        controller.wanderEnabled = settings.wanderEnabled
        controllers[slot] = controller

        guard let image = assets.cardArt(for: printing) ?? crop else {
            slots[slot]?.placement = .waitingForSurface
            return
        }
        let imageName = "slot\(slot)-\(generation)"
        do {
            try await session.track(image, name: imageName)
            guard selectionGeneration[slot] == generation else { return }
            anchor.imageName = imageName
        } catch {
            guard selectionGeneration[slot] == generation else { return }
            banner = "That card image is hard to track, so place it on a surface instead"
            slots[slot]?.placement = .waitingForSurface
        }
    }

    private func clear(slot: Int) {
        if battle.phase != .waiting {
            battle.reset()
            for controller in controllers.values { controller.opponent = nil }
        }
        if let controller = controllers.removeValue(forKey: slot) { spawner.despawn(controller) }
        if let anchor = anchors[slot] {
            if let name = anchor.imageName { session.stopTracking(name: name) }
            anchor.reset()
        }
        slots[slot] = nil
    }

    private func anchor(for slot: Int) -> CardAnchor {
        if let existing = anchors[slot] { return existing }
        let anchor = CardAnchor(parent: session.worldRoot)
        anchors[slot] = anchor
        return anchor
    }

    // MARK: Placement

    private func handleImageAnchor(_ imageAnchor: ARImageAnchor) {
        for (slot, anchor) in anchors where anchor.imageName == imageAnchor.referenceImage.name {
            let isFirstSighting = !anchor.isPlaced
            anchor.follow(imageAnchor.transform, smoothing: Float(settings.anchorSmoothing))
            if isFirstSighting { didPlace(slot: slot) }
        }
    }

    /// Taps place a waiting character on a surface; in solo, tapping the character attacks.
    func handleTap(at point: CGPoint) {
        if let pending = (0..<slotCount).first(where: { slots[$0] != nil && slots[$0]?.placement != .placed }),
           let transform = session.raycastSurface(at: point), let anchor = anchors[pending] {
            if let name = anchor.imageName {
                session.stopTracking(name: name)
                anchor.imageName = nil
            }
            anchor.place(at: transform)
            didPlace(slot: pending)
            return
        }
        if mode == .solo, let entity = session.entity(at: point), let slot = slot(owning: entity) {
            Task { await controllers[slot]?.attack() }
        }
    }

    private func didPlace(slot: Int) {
        slots[slot]?.placement = .placed
        banner = nil
        guard let controller = controllers[slot] else { return }
        Task {
            await controller.playEntrance()
            controller.settle()
        }
        if mode == .battle, allSlotsPlaced { startBattle() }
    }

    private func slot(owning entity: Entity) -> Int? {
        var current: Entity? = entity
        while let node = current {
            if let match = controllers.first(where: { $0.value.root === node }) { return match.key }
            current = node.parent
        }
        return nil
    }

    // MARK: Solo actions

    func attack(_ attack: Attack? = nil) {
        guard let controller = controllers[0] else { return }
        Task { await controller.attack(attack) }
    }

    func hit() {
        guard let controller = controllers[0] else { return }
        Task { await controller.takeHit() }
    }

    func victory() {
        guard let controller = controllers[0] else { return }
        Task { await controller.celebrate() }
    }

    // MARK: Battle

    private func startBattle() {
        guard let left = slots[0], let right = slots[1],
              let leftController = controllers[0], let rightController = controllers[1] else { return }
        leftController.opponent = rightController
        rightController.opponent = leftController
        Task {
            // Let both entrances finish before turning to face each other.
            try? await Task.sleep(for: .seconds(0.5))
            await leftController.faceOpponent()
            await rightController.faceOpponent()
            leftController.settle()
            rightController.settle()
        }
        battle.start(left: (left.card, leftController), right: (right.card, rightController))
    }

    // MARK: Scanning

    func startScan() {
        guard canScan else {
            banner = "Recognition needs card art or printings.f32 in the bundle. Pick a card instead."
            showingPicker = true
            return
        }
        identified = nil
        scanState = .searching
        session.onFrame = { [weak self] frame in self?.consider(frame) }
    }

    func stopScan() {
        scanState = .off
        debugAttempt = nil
        session.onFrame = nil
    }

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

    // MARK: Catalog display

    func isInRoster(_ printingID: String) -> Bool {
        catalog.printing(id: printingID) != nil
    }

    /// "Monkey.D.Luffy · parallel" for any catalog printing.
    func displayName(for printingID: String) -> String {
        guard let entry = fullCatalog.entry(id: printingID) else { return printingID }
        return "\(entry.name) · \(entry.kind)"
    }

    /// Art for any catalog printing: bundled roster art when there is some, otherwise the cached or
    /// downloaded thumbnail. `nil` offline or when the printing has no art URL.
    func thumbnail(for printingID: String) async -> CGImage? {
        if let printing = catalog.printing(id: printingID), let art = assets.cardArt(for: printing) { return art }
        guard let entry = fullCatalog.entry(id: printingID), let data = await thumbnails.data(for: entry),
              let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
