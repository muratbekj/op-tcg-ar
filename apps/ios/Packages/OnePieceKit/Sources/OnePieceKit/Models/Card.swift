import Foundation

/// One entry per card number (e.g. "OP05-119"), holding TCG stats. Mirrors `data/cards/cards.json`.
public struct Card: Codable, Hashable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case leader, character, event, stage
    }

    public let id: String
    public let set: String
    public let number: String
    public let name: String
    public let kind: Kind
    public let colors: [String]
    public let cost: Int?
    public let power: Int?
    public let counter: Int?
    public let life: Int?
    public let characterId: String
    public let defaultVariantId: String

    public init(
        id: String, set: String, number: String, name: String, kind: Kind,
        colors: [String] = [], cost: Int? = nil, power: Int? = nil, counter: Int? = nil,
        life: Int? = nil, characterId: String, defaultVariantId: String
    ) {
        self.id = id
        self.set = set
        self.number = number
        self.name = name
        self.kind = kind
        self.colors = colors
        self.cost = cost
        self.power = power
        self.counter = counter
        self.life = life
        self.characterId = characterId
        self.defaultVariantId = defaultVariantId
    }
}
