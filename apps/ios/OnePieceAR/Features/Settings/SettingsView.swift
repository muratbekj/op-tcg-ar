import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var scanCount = 0

    var body: some View {
        @Bindable var settings = model.settings
        NavigationStack {
            Form {
                Section("Character") {
                    Toggle("Wander around the card", isOn: $settings.wanderEnabled)
                }
                Section {
                    VStack(alignment: .leading) {
                        Text("Anchor smoothing: \(settings.anchorSmoothing, format: .number.precision(.fractionLength(2)))")
                        Slider(value: $settings.anchorSmoothing, in: 0...0.9)
                    }
                } header: {
                    Text("Tracking")
                } footer: {
                    Text("Higher smoothing reduces jitter on glossy and foil cards but lags when the card moves.")
                }
                Section {
                    LabeledContent("References", value: model.recognitionSummary ?? "None")
                    Toggle("Log scans", isOn: $settings.logScans)
                    Toggle("Show recognition debug", isOn: $settings.showDebug)
                    LabeledContent("Logged scans", value: "\(scanCount)")
                    Button("Delete scan logs", role: .destructive) {
                        Task {
                            try? await model.scanLog.deleteAll()
                            scanCount = await model.scanLog.count()
                        }
                    }
                    .disabled(scanCount == 0)
                } header: {
                    Text("Recognition")
                } footer: {
                    Text("Scan logs live in Documents/Scans. Confirm or correct a scan (the ✓ on the result strip, or the printing list) to label it for training. Copy the folder to the Mac from Finder (device > Files > OnePieceAR).")
                }
                Section {
                    Text("Personal project. Character designs and card art belong to Eiichiro Oda / Shueisha / Bandai. Nothing here is for distribution.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .task { scanCount = await model.scanLog.count() }
        }
    }
}
