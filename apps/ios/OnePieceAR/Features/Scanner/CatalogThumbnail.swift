import SwiftUI

/// Card art for any catalog printing, loaded on demand. Shows the placeholder until it loads, and
/// keeps showing it offline.
struct CatalogThumbnail: View {
    @Environment(AppModel.self) private var model
    let printingID: String
    @State private var image: CGImage?

    var body: some View {
        CardArtThumbnail(image: image)
            .task(id: printingID) {
                image = nil
                image = await model.thumbnail(for: printingID)
            }
    }
}
