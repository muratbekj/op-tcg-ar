import ARKit
import CardVision
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
    private(set) var loadError: String?
    private(set) var mode: Mode = .solo
    private(set) var slots: [Int: Slot] = [:]
    /// Which slot the next scan or pick fills.
    private(set) var targetSlot = 0
    private(set) var scanState: ScanState = .off
    /// The most recent recognition, kept for the "not this one?" list.
    private(set) var lastRecognition: (slot: Int, result: RecognitionResult)?
    private(set) var recognitionSummary: String?
    private(set) var banner: String?

    var showingPicker = false
    var showingAlternatives = false
    var showingSettings = false

    let settings = AppSettings()
    let session = ARSessionManager()
    let battle = BattleCoordinator()
    @ObservationIgnored let assets = AssetService()
    @ObservationIgnored let scanLog = ScanLogger()
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
        let fullCatalog: FullCatalog
        if let url = Bundle.main.url(forResource: "catalog", withExtension: "json"), let loaded = try? FullCatalog.load(from: url) {
            fullCatalog = loaded
        } else {
            print("AppModel: no catalog.json bundled; recognizing roster printings only")
            fullCatalog = FullCatalog(roster: catalog)
        }
        let bundle = Bundle.main
        if await recognition.prepare(
            catalog: catalog,
            fullCatalog: fullCatalog,
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
        lastRecognition = nil
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
    func select(_ printing: Printing, slot requestedSlot: Int? = nil, crop: CGImage? = nil, scanID: String? = nil) async {
        let slot = requestedSlot ?? targetSlot
        guard let card = catalog.card(for: printing), let variant = catalog.variant(for: printing) else {
            banner = "No character variant for \(printing.id)"
            return
        }
        let generation = selectionGeneration[slot, default: 0] + 1
        selectionGeneration[slot] = generation

        clear(slot: slot)
        if scanID == nil, lastRecognition?.slot == slot { lastRecognition = nil }
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
        scanState = .searching
        session.onFrame = { [weak self] frame in self?.consider(frame) }
    }

    func stopScan() {
        scanState = .off
        session.onFrame = nil
    }

    private func consider(_ frame: ARFrame) {
        let now = Date.now
        guard !recognitionBusy, now.timeIntervalSince(lastRecognitionAttempt) > Self.recognitionInterval else { return }
        recognitionBusy = true
        lastRecognitionAttempt = now
        let buffer = PixelBufferBox(buffer: frame.capturedImage)
        Task {
            let result = try? await recognition.recognize(buffer)
            recognitionBusy = false
            guard scanState == .searching, let result, let best = result.best,
                  let printing = catalog.printing(id: best.printingID) else { return }
            stopScan()
            let slot = targetSlot
            let scanID = settings.logScans ? try? await scanLog.log(result, spawnedPrintingID: printing.id) : nil
            lastRecognition = (slot, result)
            await select(printing, slot: slot, crop: result.crop, scanID: scanID)
        }
    }

    /// "Not this one?": swap to another ranked candidate and record the correction.
    func correct(to candidate: RecognitionCandidate) async {
        guard let (slot, result) = lastRecognition, let printing = catalog.printing(id: candidate.printingID) else { return }
        let scanID = slots[slot]?.scanID
        if let scanID {
            try? await scanLog.markCorrected(scanID: scanID, finalPrintingID: printing.id)
        }
        await select(printing, slot: slot, crop: result.crop, scanID: scanID)
    }
}
