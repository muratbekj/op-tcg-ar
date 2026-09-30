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
