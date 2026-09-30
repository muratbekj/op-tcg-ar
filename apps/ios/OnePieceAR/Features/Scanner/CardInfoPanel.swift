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
