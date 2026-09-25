import OnePieceKit
import SwiftUI

struct CardDetailView: View {
    @Environment(AppModel.self) private var model
    let printing: Printing
    let onSummon: (Printing) -> Void

    var body: some View {
        let card = model.catalog.card(for: printing)
        let variant = model.catalog.variant(for: printing)
        List {
            Section {
                CardArtThumbnail(image: model.assets.cardArt(for: printing))
                    .frame(maxHeight: 260)
                    .frame(maxWidth: .infinity)
                    .listRowBackground(Color.clear)
            }
            if let card {
                Section("Card") {
                    LabeledContent("Number", value: card.id)
                    LabeledContent("Kind", value: card.kind.rawValue.capitalized)
                    LabeledContent("Colors", value: card.colors.map(\.capitalized).joined(separator: ", "))
                    if let cost = card.cost { LabeledContent("Cost", value: "\(cost)") }
                    if let power = card.power { LabeledContent("Power", value: "\(power)") }
                    if let counter = card.counter { LabeledContent("Counter", value: "\(counter)") }
                    if let life = card.life { LabeledContent("Life", value: "\(life)") }
                }
            }
            Section("Printing") {
                LabeledContent("ID", value: printing.id)
                LabeledContent("Kind", value: printing.kind.rawValue.capitalized)
                LabeledContent("Rarity", value: printing.rarity)
                LabeledContent("Variant source", value: printing.variantId == nil ? "Card default" : "Printing override")
                AssetBadges(printing: printing)
            }
            if let variant {
                Section("Variant: \(variant.name)") {
                    LabeledContent("Model", value: variant.modelAsset)
                    LabeledContent("Height", value: "\(Int(variant.heightMeters * 100)) cm")
                    ForEach(variant.animations.allClips, id: \.self) { clip in
                        LabeledContent(clip) {
                            let found = model.assets.clipURL(clip, for: variant) != nil
                            Image(systemName: found ? "checkmark.circle.fill" : "circle.dashed")
                                .foregroundStyle(found ? .green : .secondary)
                        }
                    }
                    ForEach(variant.animations.attacks) { attack in
                        LabeledContent(attack.displayName, value: attack.vfx.map(\.rawValue).joined(separator: " + "))
                    }
                }
            }
            Section {
                Button("Summon") { onSummon(printing) }
                    .frame(maxWidth: .infinity)
                    .disabled(variant == nil)
            }
        }
        .navigationTitle(card?.name ?? printing.id)
        .navigationBarTitleDisplayMode(.inline)
    }
}
