import Foundation

/// A character across all its forms. Not stored on disk: derived from cards and variants
/// that share a `characterId`.
public struct Character: Hashable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let variantIds: [String]

    public init(id: String, name: String, variantIds: [String]) {
        self.id = id
        self.name = name
        self.variantIds = variantIds
    }
}
