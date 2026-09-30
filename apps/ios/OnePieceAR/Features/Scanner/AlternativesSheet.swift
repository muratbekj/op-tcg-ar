import CardVision
import OnePieceKit
import SwiftUI

/// "Not this one?": the ranked alternatives from the last scan.
struct AlternativesSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let recognition = model.lastRecognition {
                    Section {
                        HStack {
                            Image(decorative: recognition.result.crop, scale: 1)
                                .resizable()
                                .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
                                .frame(height: 110)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            VStack(alignment: .leading) {
                                Text("What the camera saw").font(.headline)
                                Text("Card number read: \(recognition.result.ocrCardID ?? "none")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    Section("Ranked by art similarity") {
                        ForEach(recognition.result.candidates) { candidate in
                            candidateRow(candidate, isCurrent: model.slots[recognition.slot]?.printing.id == candidate.printingID)
                        }
                    }
                }
                Section {
                    Button("None of these, pick manually") {
                        dismiss()
                        model.showingPicker = true
                    }
                }
            }
            .navigationTitle("Not this one?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }

    private func candidateRow(_ candidate: RecognitionCandidate, isCurrent: Bool) -> some View {
        let printing = model.catalog.printing(id: candidate.printingID)
        let card = model.catalog.card(id: candidate.cardID)
        let variant = printing.flatMap { model.catalog.variant(for: $0) }
        return Button {
            dismiss()
            Task { await model.correct(to: candidate) }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(card?.name ?? candidate.cardID) · \(variant?.name ?? "?")").font(.body.weight(.medium))
                    Text("\(candidate.printingID) · \(printing?.kind.rawValue ?? "") \(printing?.rarity ?? "")")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if candidate.cardID == model.lastRecognition?.result.ocrCardID {
                    Image(systemName: "number").foregroundStyle(.blue).accessibilityLabel("Card number matches")
                }
                if let similarity = candidate.similarity {
                    Text(similarity, format: .percent.precision(.fractionLength(0)))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if isCurrent {
                    Image(systemName: "checkmark").foregroundStyle(.tint)
                }
            }
        }
        .tint(.primary)
    }
}
