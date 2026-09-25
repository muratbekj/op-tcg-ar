import Testing
import OnePieceKit
@testable import BattleKit

@Suite struct BattleEngineTests {
    func fighter(_ name: String, power: Int, counter: Int = 0, life: Int = 2) -> Fighter {
        Fighter(name: name, cardID: name, power: power, counter: counter, maxLife: life)
    }

    @Test func equalPowerLands() throws {
        var engine = BattleEngine(left: fighter("A", power: 5000), right: fighter("B", power: 5000))
        let outcome = try engine.attack(from: .left)
        #expect(outcome.landed)
        #expect(engine.right.life == 1)
        #expect(engine.turn == .right)
    }

    @Test func weakerAttackMisses() throws {
        var engine = BattleEngine(left: fighter("A", power: 4000), right: fighter("B", power: 5000))
        #expect(try !engine.attack(from: .left).landed)
        #expect(engine.right.life == 2)
    }

    @Test func donTapsBoostAndClamp() throws {
        var engine = BattleEngine(left: fighter("A", power: 4000), right: fighter("B", power: 6000))
        let outcome = try engine.attack(from: .left, donTaps: 5)
        #expect(outcome.donTaps == 2)
        #expect(outcome.attackPower == 6000)
        #expect(outcome.landed)
    }

    @Test func counterBlocksAndIsLimitedPerMatch() throws {
        var engine = BattleEngine(left: fighter("A", power: 5000, life: 9), right: fighter("B", power: 5000, counter: 1000, life: 9))
        for _ in 0..<BattleRules.countersPerMatch {
            let outcome = try engine.attack(from: .left)
            #expect(outcome.counterUsed)
            #expect(!outcome.landed)
            try engine.attack(from: .right)
        }
        let third = try engine.attack(from: .left)
        #expect(!third.counterUsed)
        #expect(third.landed)
    }

    @Test func counterNotWastedWhenItCannotBlock() throws {
        var engine = BattleEngine(left: fighter("A", power: 7000), right: fighter("B", power: 5000, counter: 1000))
        let outcome = try engine.attack(from: .left)
        #expect(!outcome.counterUsed)
        #expect(outcome.landed)
        #expect(engine.right.countersLeft == BattleRules.countersPerMatch)
    }

    @Test func knockoutEndsBattle() throws {
        var engine = BattleEngine(left: fighter("A", power: 5000), right: fighter("B", power: 1000, life: 1))
        let outcome = try engine.attack(from: .left)
        #expect(outcome.knockout)
        #expect(engine.winner == .left)
        #expect(throws: BattleError.battleOver) { try engine.attack(from: .right) }
    }

    @Test func enforcesTurnOrder() {
        var engine = BattleEngine(left: fighter("A", power: 1), right: fighter("B", power: 1))
        #expect(throws: BattleError.notYourTurn) { try engine.attack(from: .right) }
    }

    @Test func derivedLife() {
        func card(_ kind: Card.Kind, cost: Int? = nil, life: Int? = nil) -> Card {
            Card(id: "x", set: "x", number: "x", name: "x", kind: kind, cost: cost, life: life, characterId: "x", defaultVariantId: "x")
        }
        #expect(BattleRules.life(for: card(.leader, life: 4)) == 4)
        #expect(BattleRules.life(for: card(.leader)) == BattleRules.fallbackLeaderLife)
        #expect(BattleRules.life(for: card(.character, cost: 1)) == 2)
        #expect(BattleRules.life(for: card(.character, cost: 4)) == 3)
        #expect(BattleRules.life(for: card(.character, cost: 10)) == 4)
    }
}
