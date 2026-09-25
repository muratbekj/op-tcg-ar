// swift-tools-version: 6.0
import PackageDescription

// Pure Swift logic shared by the app: data model, catalog, recognition math, battle rules.
// No ARKit/RealityKit/Vision imports, so everything here runs under `swift test` on the Mac.
let package = Package(
    name: "OnePieceKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "OnePieceKit", targets: ["OnePieceKit"]),
        .library(name: "BattleKit", targets: ["BattleKit"]),
    ],
    targets: [
        .target(name: "OnePieceKit"),
        .target(name: "BattleKit", dependencies: ["OnePieceKit"]),
        .testTarget(name: "OnePieceKitTests", dependencies: ["OnePieceKit"]),
        .testTarget(name: "BattleKitTests", dependencies: ["BattleKit"]),
    ]
)
