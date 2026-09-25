import OnePieceKit

/// TCG-flavored rules kept close to the real game: an attack lands when attacking power is at
/// least the defender's power. Effect text is out of scope for now.
public enum BattleRules {
    public static let donBoostPerTap = 1000
    public static let maxDonTaps = 2
    public static let countersPerMatch = 2
    /// Used when a leader card has no printed life.
    public static let fallbackLeaderLife = 5

    /// Leaders use printed life. Characters get a small value that grows slowly with cost:
    /// cost 0-2 -> 2, 3-5 -> 3, 6+ -> 4.
    public static func life(for card: Card) -> Int {
        switch card.kind {
        case .leader:
            return card.life ?? fallbackLeaderLife
        case .character, .event, .stage:
            return min(4, max(2, 2 + (card.cost ?? 0) / 3))
        }
    }
}
