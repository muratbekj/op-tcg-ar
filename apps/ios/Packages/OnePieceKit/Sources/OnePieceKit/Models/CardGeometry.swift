import Foundation

/// Physical dimensions of a standard One Piece card (63 x 88 mm).
public enum CardGeometry {
    public static let widthMeters = 0.063
    public static let heightMeters = 0.088
    /// Short side over long side, about 0.716. Vision reports rectangle aspect this way.
    public static let aspectRatio = widthMeters / heightMeters
}
