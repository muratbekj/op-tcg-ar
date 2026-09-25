import OnePieceKit
import SwiftUI

/// Manual card picker and roster browser: every printing, the variant it resolves to, and which
/// assets are bundled.
struct CollectionView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    let onSummon: (Printing) -> Void

    var body: some View {
        NavigationStack {
            List {
                ForEach(model.catalog.cards) { card in
                    Section("\(card.id) · \(card.name)") {
                        ForEach(model.catalog.printings(ofCard: card.id)) { printing in
                            PrintingRow(printing: printing, onSummon: onSummon)
                        }
                    }
                }
            }
            .overlay {
                if model.catalog.cards.isEmpty {
                    ContentUnavailableView("No cards", systemImage: "rectangle.stack",
                                           description: Text("data/cards/cards.json is empty or missing from the bundle."))
                }
            }
            .navigationTitle("Cards")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: String.self) { printingID in
                if let printing = model.catalog.printing(id: printingID) {
                    CardDetailView(printing: printing, onSummon: onSummon)
                }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
    }
}

private struct PrintingRow: View {
    @Environment(AppModel.self) private var model
    let printing: Printing
    let onSummon: (Printing) -> Void

    var body: some View {
        let variant = model.catalog.variant(for: printing)
        HStack(spacing: 12) {
            CardArtThumbnail(image: model.assets.cardArt(for: printing))
                .frame(width: 44)
            VStack(alignment: .leading, spacing: 3) {
                Text(variant?.name ?? "No variant").font(.body.weight(.semibold))
                Text("\(printing.kind.rawValue.capitalized) · \(printing.rarity) · \(printing.language.uppercased())")
                    .font(.caption).foregroundStyle(.secondary)
                AssetBadges(printing: printing)
            }
            Spacer()
            Button("Summon") { onSummon(printing) }
                .buttonStyle(.borderedProminent)
                .disabled(variant == nil)
            NavigationLink(value: printing.id) {
                Image(systemName: "info.circle")
            }
            .fixedSize()
        }
        .buttonStyle(.borderless)
    }
}

struct AssetBadges: View {
    @Environment(AppModel.self) private var model
    let printing: Printing

    var body: some View {
        let hasModel = model.catalog.variant(for: printing).map(model.assets.hasModel(for:)) ?? false
        let hasArt = model.assets.cardArt(for: printing) != nil
        HStack(spacing: 6) {
            badge(hasModel ? "3D model" : "placeholder", ok: hasModel)
            badge(hasArt ? "art" : "no art", ok: hasArt)
        }
    }

    private func badge(_ text: String, ok: Bool) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background((ok ? Color.green : Color.orange).opacity(0.2), in: Capsule())
    }
}

struct CardArtThumbnail: View {
    let image: CGImage?

    var body: some View {
        Group {
            if let image {
                Image(decorative: image, scale: 1).resizable()
            } else {
                Rectangle().fill(.quaternary)
                    .overlay { Image(systemName: "photo").foregroundStyle(.secondary) }
            }
        }
        .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }
}
