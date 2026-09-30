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
