// swift-tools-version: 6.0
import PackageDescription

// Shared by the app and the ML lab:
// - OnePieceKit: data model, catalog, recognition math. No ARKit/RealityKit/Vision.
// - BattleKit: battle rules. Pure Swift.
// - CardVision: Vision/Core ML recognition pipeline (runs on iOS and macOS).
// - CardVisionCLI (`cardvision`): Mac CLI that runs CardVision for reference embeddings and offline evaluation.
// Everything here runs under `swift test` on the Mac.
let package = Package(
    name: "OnePieceKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "OnePieceKit", targets: ["OnePieceKit"]),
        .library(name: "BattleKit", targets: ["BattleKit"]),
        .library(name: "CardVision", targets: ["CardVision"]),
        .executable(name: "cardvision", targets: ["CardVisionCLI"]),
    ],
    targets: [
        .target(name: "OnePieceKit"),
        .target(name: "BattleKit", dependencies: ["OnePieceKit"]),
        .target(name: "CardVision", dependencies: ["OnePieceKit"]),
        .executableTarget(name: "CardVisionCLI", dependencies: ["CardVision", "OnePieceKit"]),
        .testTarget(name: "OnePieceKitTests", dependencies: ["OnePieceKit"]),
        .testTarget(name: "BattleKitTests", dependencies: ["BattleKit"]),
        .testTarget(name: "CardVisionTests", dependencies: ["CardVision"]),
    ]
)
