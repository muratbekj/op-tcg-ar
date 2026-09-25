import SwiftUI

@main
struct OnePieceARApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ExperienceView()
                .environment(model)
        }
    }
}
