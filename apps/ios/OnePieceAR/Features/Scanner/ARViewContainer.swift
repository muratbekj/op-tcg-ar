import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    let model: AppModel

    func makeUIView(context: Context) -> ARView {
        let view = model.session.arView
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        view.addGestureRecognizer(tap)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    final class Coordinator: NSObject {
        let model: AppModel

        init(model: AppModel) {
            self.model = model
        }

        @objc func tapped(_ gesture: UITapGestureRecognizer) {
            model.handleTap(at: gesture.location(in: gesture.view))
        }
    }
}
