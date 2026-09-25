import Foundation

/// One character form (e.g. Gear 4 Luffy) and the assets that bring it to life.
/// Mirrors `data/cards/variants.json`.
public struct CharacterVariant: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let characterId: String
    public let name: String
    /// USDZ file name in the app bundle, e.g. "luffy_gear4.usdz".
    public let modelAsset: String
    /// Target standing height in the AR scene. A card is 0.088 m tall, for scale.
    public let heightMeters: Double
    public let animations: AnimationSet

    public init(
        id: String, characterId: String, name: String, modelAsset: String,
        heightMeters: Double, animations: AnimationSet
    ) {
        self.id = id
        self.characterId = characterId
        self.name = name
        self.modelAsset = modelAsset
        self.heightMeters = heightMeters
        self.animations = animations
    }
}

/// Clip names refer to the shared animation library; a variant declares which ones it supports.
public struct AnimationSet: Codable, Hashable, Sendable {
    public let idle: String
    public let walk: String?
    public let hit: String
    public let ko: String?
    public let victory: String?
    public let attacks: [Attack]

    public init(
        idle: String, walk: String? = nil, hit: String, ko: String? = nil,
        victory: String? = nil, attacks: [Attack]
    ) {
        self.idle = idle
        self.walk = walk
        self.hit = hit
        self.ko = ko
        self.victory = victory
        self.attacks = attacks
    }

    /// Every distinct clip name this set references, idle first.
    public var allClips: [String] {
        var seen = Set<String>()
        return ([idle, walk, hit, ko, victory].compactMap { $0} + attacks.map(\.clip))
            .filter { seen.insert($0).inserted }
    }
}

public struct Attack: Codable, Hashable, Identifiable, Sendable {
    public let id: String
    public let clip: String
    public let vfx: [VFXKind]

    public init(id: String, clip: String, vfx: [VFXKind] = []) {
        self.id = id
        self.clip = clip
        self.vfx = vfx
    }

    /// "kong_gun" -> "Kong Gun"
    public var displayName: String {
        id.split(separator: "_").map { $0.capitalized }.joined(separator: " ")
    }
}

/// Effects live apart from models so a new character composes existing ones.
public enum VFXKind: String, Codable, CaseIterable, Sendable {
    case haki, slash, fire, lightning, smoke, impact
}
