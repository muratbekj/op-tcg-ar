import Foundation

/// User-tunable settings, persisted in UserDefaults.
@Observable
final class AppSettings {
    private enum Key {
        static let wander = "wanderEnabled"
        static let smoothing = "anchorSmoothing"
        static let logScans = "logScans"
        static let debug = "showDebug"
    }

    var wanderEnabled: Bool { didSet { defaults.set(wanderEnabled, forKey: Key.wander) } }
    /// 0 snaps to every ARKit update; higher trades lag for less jitter on foil cards.
    var anchorSmoothing: Double { didSet { defaults.set(anchorSmoothing, forKey: Key.smoothing) } }
    var logScans: Bool { didSet { defaults.set(logScans, forKey: Key.logScans) } }
    var showDebug: Bool { didSet { defaults.set(showDebug, forKey: Key.debug) } }

    @ObservationIgnored private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        defaults.register(defaults: [Key.wander: true, Key.smoothing: 0.6, Key.logScans: true, Key.debug: false])
        wanderEnabled = defaults.bool(forKey: Key.wander)
        anchorSmoothing = defaults.double(forKey: Key.smoothing)
        logScans = defaults.bool(forKey: Key.logScans)
        showDebug = defaults.bool(forKey: Key.debug)
    }
}
