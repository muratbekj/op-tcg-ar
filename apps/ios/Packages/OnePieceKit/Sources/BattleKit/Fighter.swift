import OnePieceKit

public enum Side: String, CaseIterable, Sendable {
    case left, right

    public var opponent: Side { self == .left ? .right : .left }
}

public struct Fighter: Equatable, Sendable {
    public let name: String
    public let cardID: String
    public let power: Int
    public let counter: Int
    public let maxLife: Int
    public internal(set) var life: Int
    public internal(set) var countersLeft: Int

    public init(name: String, cardID: String, power: Int, counter: Int, maxLife: Int) {
        self.name = name
        self.cardID = cardID
        self.power = power
        self.counter = counter
        self.maxLife = maxLife
        life = maxLife
        countersLeft = counter > 0 ? BattleRules.countersPerMatch : 0
    }

    public init(card: Card) {
        self.init(
            name: card.name,
            cardID: card.id,
            power: card.power ?? 0,
            counter: card.counter ?? 0,
            maxLife: BattleRules.life(for: card))
    }

    public var isKnockedOut: Bool { life <= 0 }
}
