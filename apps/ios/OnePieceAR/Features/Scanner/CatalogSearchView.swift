import OnePieceKit
import SwiftUI

/// Search the whole catalog (not just the roster) to name the printing a scan really was.
struct CatalogSearchView: View {
    @Environment(AppModel.self) private var model
    let onPick: (String) -> Void
    @State private var query: String

    private static let minimumQueryLength = 2
    private static let maxRows = 100

    init(initialQuery: String = "", onPick: @escaping (String) -> Void) {
        self.onPick = onPick
        _query = State(initialValue: initialQuery)
    }

    private var results: [CatalogEntry] {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard q.count >= Self.minimumQueryLength else { return [] }
        return model.fullCatalog.entries
            .lazy
            .filter {
                $0.printingId.localizedCaseInsensitiveContains(q)
                    || $0.cardId.localizedCaseInsensitiveContains(q)
                    || $0.name.localizedCaseInsensitiveContains(q)
            }
            .prefix(Self.maxRows)
            .map { $0 }
    }

    var body: some View {
        let rows = results
        List {
            if query.trimmingCharacters(in: .whitespaces).count < Self.minimumQueryLength {
                Text("Type a printing ID, card code, or name.").foregroundStyle(.secondary)
            } else if rows.isEmpty {
                Text("No matching printings.").foregroundStyle(.secondary)
            }
            ForEach(rows) { entry in
                Button { onPick(entry.printingId) } label: {
                    HStack(spacing: 12) {
                        CatalogThumbnail(printingID: entry.printingId)
                            .frame(width: 44)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.name).font(.body.weight(.medium))
                            Text([entry.printingId, entry.kind, entry.rarity].filter { !$0.isEmpty }.joined(separator: " · "))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .tint(.primary)
            }
        }
        .navigationTitle("Find the card")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "ID, code, or name")
    }
}
