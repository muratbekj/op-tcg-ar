/// Turn-based duel between two fighters. Pure value type: no AR, no timers, fully unit-testable.
public struct BattleEngine: Sendable {
    public private(set) var left: Fighter
    public private(set) var right: Fighter
    public private(set) var turn: Side
    public private(set) var winner: Side?
    public private(set) var log: [AttackOutcome] = []

    public init(left: Fighter, right: Fighter, firstTurn: Side = .left) {
        self.left = left
        self.right = right
        turn = firstTurn
    }

    public subscript(side: Side) -> Fighter {
        get { side == .left ? left : right }
        set {
            if side == .left { left = newValue } else { right = newValue }
        }
    }

    public var isOver: Bool { winner != nil }

    /// Resolves one attack from `side`, then passes the turn.
    /// - Parameter donTaps: timing taps landed during the charge, clamped to `0...maxDonTaps`.
    @discardableResult
    public mutating func attack(
        from side: Side, donTaps: Int = 0, counterPolicy: CounterPolicy = .whenItBlocks
    ) throws(BattleError) -> AttackOutcome {
        guard winner == nil else { throw .battleOver }
        guard side == turn else { throw .notYourTurn }

        let taps = min(max(donTaps, 0), BattleRules.maxDonTaps)
        let attackPower = self[side].power + taps * BattleRules.donBoostPerTap
        var defender = self[side.opponent]

        let landsUncountered = attackPower >= defender.power
        let counterBlocks = attackPower < defender.power + defender.counter
        let counterUsed = counterPolicy == .whenItBlocks
            && defender.countersLeft > 0 && landsUncountered && counterBlocks
        if counterUsed { defender.countersLeft -= 1 }

        let defensePower = defender.power + (counterUsed ? defender.counter : 0)
        let landed = attackPower >= defensePower
        if landed { defender.life = max(0, defender.life - 1) }
        self[side.opponent] = defender

        let outcome = AttackOutcome(
            attacker: side, attackPower: attackPower, defensePower: defensePower, donTaps: taps,
            counterUsed: counterUsed, landed: landed, defenderLife: defender.life,
            knockout: defender.isKnockedOut)
        log.append(outcome)

        if defender.isKnockedOut {
            winner = side
        } else {
            turn = side.opponent
        }
        return outcome
    }
}

public enum CounterPolicy: Sendable {
    case never
    /// Spend a counter only when the hit would land without it and the counter stops it.
    case whenItBlocks
}

public struct AttackOutcome: Equatable, Sendable {
    public let attacker: Side
    public let attackPower: Int
    public let defensePower: Int
    public let donTaps: Int
    public let counterUsed: Bool
    public let landed: Bool
    public let defenderLife: Int
    public let knockout: Bool
}

public enum BattleError: Error, Equatable {
    case notYourTurn
    case battleOver
}
