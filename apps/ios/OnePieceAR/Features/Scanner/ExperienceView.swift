import CardVision
import OnePieceKit
import SwiftUI

/// The main screen: camera, status, and mode-specific controls.
struct ExperienceView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        @Bindable var model = model
        ZStack {
            if ARSessionManager.isSupported {
                ARViewContainer(model: model).ignoresSafeArea()
            } else {
                UnsupportedDeviceView()
            }

            if model.scanState == .searching {
                ScanGuide()
            }

            VStack(spacing: 12) {
                TopBar()
                if let status = model.statusText {
                    StatusPill(text: status)
                }
                if model.settings.showDebug, let recognition = model.lastRecognition {
                    RecognitionDebugView(result: recognition.result)
                }
                Spacer()
                bottomControls
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .sheet(isPresented: $model.showingPicker) {
            CollectionView { printing in
                model.showingPicker = false
                Task { await model.select(printing) }
            }
        }
        .sheet(isPresented: $model.showingAlternatives) {
            AlternativesSheet()
                .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $model.showingSettings) {
            SettingsView()
        }
        .alert("Card data failed to load", isPresented: .constant(model.loadError != nil)) {
        } message: {
            Text(model.loadError ?? "")
        }
        .task { await model.bootstrap() }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active: model.session.start()
            case .background: model.session.pause()
            default: break
            }
        }
        .onChange(of: model.settings.wanderEnabled) { model.applySettings() }
    }

    @ViewBuilder private var bottomControls: some View {
        if model.lastRecognition != nil, model.scanState == .off {
            Button("Not this one?") { model.showingAlternatives = true }
                .buttonStyle(.bordered)
                .tint(.white)
        }
        switch model.mode {
        case .solo:
            if let slot = model.soloSlot, slot.placement == .placed {
                SoloActionBar(slot: slot)
            } else {
                SummonBar(prompt: nil)
            }
        case .battle:
            if model.allSlotsPlaced, model.battle.phase != .waiting {
                BattleHUD()
            } else {
                SummonBar(prompt: "Player \(model.targetSlot + 1): scan or pick a card")
            }
        }
    }
}

private struct TopBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack {
            Picker("Mode", selection: Binding(get: { model.mode }, set: { model.setMode($0) })) {
                Text("Solo").tag(AppModel.Mode.solo)
                Text("Battle").tag(AppModel.Mode.battle)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 200)

            Spacer()

            if !model.slots.isEmpty {
                CircleButton(systemImage: "arrow.counterclockwise", label: "Reset") { model.clearAll() }
            }
            CircleButton(systemImage: "square.grid.2x2", label: "Cards") { model.showingPicker = true }
            CircleButton(systemImage: "gearshape", label: "Settings") { model.showingSettings = true }
        }
        .padding(.top, 8)
    }
}

struct CircleButton: View {
    let systemImage: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 17, weight: .semibold))
                .frame(width: 44, height: 44)
                .background(.ultraThinMaterial, in: Circle())
        }
        .accessibilityLabel(label)
        .tint(.primary)
    }
}

struct StatusPill: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.footnote.weight(.medium))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .transition(.opacity)
    }
}

/// Scan or pick. Shown until a character is placed.
private struct SummonBar: View {
    @Environment(AppModel.self) private var model
    let prompt: String?

    var body: some View {
        VStack(spacing: 10) {
            if let prompt { StatusPill(text: prompt) }
            HStack(spacing: 12) {
                if model.scanState == .searching {
                    Button { model.stopScan() } label: {
                        Label("Stop scanning", systemImage: "xmark").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button { model.startScan() } label: {
                        Label("Scan card", systemImage: "viewfinder").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                Button { model.showingPicker = true } label: {
                    Label("Pick card", systemImage: "rectangle.stack").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .controlSize(.large)
        }
    }
}

/// Idle, attack, hit, victory for the single character.
private struct SoloActionBar: View {
    @Environment(AppModel.self) private var model
    let slot: AppModel.Slot

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(slot.card.name) · \(slot.variant.name)").font(.headline)
                    Text(slot.usesPlaceholder ? "\(slot.printing.id) · placeholder model" : slot.printing.id)
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            HStack(spacing: 10) {
                ForEach(slot.variant.animations.attacks) { attack in
                    Button(attack.displayName) { model.attack(attack) }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                }
                Button("Hit") { model.hit() }.buttonStyle(.bordered)
                Button("Victory") { model.victory() }.buttonStyle(.bordered)
            }
            .controlSize(.large)
        }
        .padding(14)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

/// Card-shaped frame to aim with while scanning.
private struct ScanGuide: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 14)
            .strokeBorder(.white.opacity(0.9), style: StrokeStyle(lineWidth: 3, dash: [14, 8]))
            .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
            .frame(width: 230)
            .shadow(radius: 6)
            .allowsHitTesting(false)
    }
}

private struct RecognitionDebugView: View {
    let result: RecognitionResult

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(decorative: result.crop, scale: 1)
                .resizable()
                .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
                .frame(width: 60)
            VStack(alignment: .leading, spacing: 2) {
                Text("OCR: \(result.ocrCardID ?? "–")  \(result.method.rawValue)  n=\(result.groupSize)")
                ForEach(result.candidates.prefix(3)) { candidate in
                    Text("\(candidate.printingID)  \(candidate.similarity.map { String(format: "%.3f", $0) } ?? "–")")
                }
            }
            .font(.caption.monospaced())
            Spacer()
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct UnsupportedDeviceView: View {
    var body: some View {
        ContentUnavailableView(
            "AR needs a real iPhone",
            systemImage: "arkit",
            description: Text("The simulator has no camera or ARKit. The card browser and settings still work."))
    }
}
