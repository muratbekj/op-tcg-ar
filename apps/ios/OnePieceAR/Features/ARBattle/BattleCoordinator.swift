import BattleKit
import Observation
import OnePieceKit

/// Drives a two-card battle: turn order, the DON!! timing window, and mapping each rules outcome
/// onto character animations. Rules live in BattleKit; this only choreographs.
@Observable
final class BattleCoordinator {
    enum Phase: Equatable {
        case waiting
        case ready
        case charging(Side)
        case resolving
        case over(winner: Side)
    }

    static let chargeWindow: Duration = .seconds(1.2)

    private(set) var engine: BattleEngine?
    private(set) var phase: Phase = .waiting
    private(set) var donTaps = 0
    private(set) var lastEvent: String?

    @ObservationIgnored private var cards: [Side: Card] = [:]
    @ObservationIgnored private var fighters: [Side: CharacterController] = [:]

    func start(left: (Card, CharacterController), right: (Card, CharacterController)) {
        cards = [.left: left.0, .right: right.0]
        fighters = [.left: left.1, .right: right.1]
        engine = BattleEngine(left: Fighter(card: left.0), right: Fighter(card: right.0))
        phase = .ready
        lastEvent = nil
    }

    func reset() {
        engine = nil
        cards = [:]
        fighters = [:]
        phase = .waiting
        lastEvent = nil
    }

    func rematch() {
        guard let left = cards[.left], let right = cards[.right] else { return }
        engine = BattleEngine(left: Fighter(card: left), right: Fighter(card: right))
        fighters.values.forEach { $0.reset() }
        phase = .ready
        lastEvent = nil
    }

    /// Opens the DON!! window for whoever's turn it is; the attack resolves when it closes.
    func beginAttack() {
        guard phase == .ready, let turn = engine?.turn else { return }
        phase = .charging(turn)
        donTaps = 0
        Task {
            try? await Task.sleep(for: Self.chargeWindow)
            await resolve()
        }
    }

    func tapDon() {
        guard case .charging = phase else { return }
        donTaps = min(donTaps + 1, BattleRules.maxDonTaps)
    }

    private func resolve() async {
        guard case .charging(let side) = phase, var engine,
              let attacker = fighters[side], let defender = fighters[side.opponent] else { return }
        phase = .resolving
        guard let outcome = try? engine.attack(from: side, donTaps: donTaps) else {
            phase = .ready
            return
        }
        self.engine = engine
        lastEvent = describe(outcome, attacker: engine[side].name)

        let lunge = Task { await attacker.attack() }
        try? await Task.sleep(for: .seconds(0.2))
        await defender.takeHit(blocked: !outcome.landed)
        await lunge.value

        if outcome.knockout {
            await defender.knockOut()
            await attacker.celebrate()
            phase = .over(winner: side)
        } else {
            phase = .ready
        }
    }

    private func describe(_ outcome: AttackOutcome, attacker: String) -> String {
        let boost = outcome.donTaps > 0 ? " (+\(outcome.donTaps * BattleRules.donBoostPerTap) DON!!)" : ""
        let power = "\(outcome.attackPower) vs \(outcome.defensePower)"
        if outcome.knockout { return "\(attacker) wins! \(power)\(boost)" }
        if outcome.counterUsed { return "Countered! \(power)\(boost)" }
        return outcome.landed ? "Hit! \(power)\(boost)" : "Blocked. \(power)\(boost)"
    }
}
