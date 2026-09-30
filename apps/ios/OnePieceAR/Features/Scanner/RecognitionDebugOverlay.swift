import CardVision
import OnePieceKit
import SwiftUI

/// The detected card's outline, drawn over the camera image for the frame recognition last looked at.
/// Settings > Show recognition debug.
struct RecognitionDebugOverlay: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        GeometryReader { geometry in
            if let quad = model.debugAttempt?.quad,
               let transform = model.session.displayTransform(viewportSize: geometry.size) {
                let points = quad.sensorCorners.map { corner in
                    let normalized = corner.applying(transform)
                    return CGPoint(x: normalized.x * geometry.size.width, y: normalized.y * geometry.size.height)
                }
                Path { path in
                    path.addLines(points)
                    path.closeSubpath()
                }
                .stroke(model.debugAttempt?.result == nil ? Color.yellow : Color.green, lineWidth: 3)
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

/// What recognition decided: OCR read, method, group size, and the top candidates' similarities.
struct RecognitionDebugView: View {
    let attempt: RecognitionAttempt?
    let lastResult: RecognitionResult?

    var body: some View {
        let result = attempt?.result ?? lastResult
        HStack(alignment: .top, spacing: 10) {
            if let result {
                Image(decorative: result.crop, scale: 1)
                    .resizable()
                    .aspectRatio(CardGeometry.aspectRatio, contentMode: .fit)
                    .frame(width: 60)
            }
            VStack(alignment: .leading, spacing: 2) {
                if let attempt, attempt.result == nil {
                    Text(attempt.quad == nil ? "No card detected" : "Card found, no confident match")
                }
                if let result {
                    Text("OCR: \(result.ocrCardID ?? "–")  \(result.method.rawValue)  n=\(result.groupSize)")
                    ForEach(result.candidates.prefix(3)) { candidate in
                        Text("\(candidate.printingID)  \(candidate.similarity.map { String(format: "%.3f", $0) } ?? "–")")
                    }
                }
            }
            .font(.caption.monospaced())
            Spacer()
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}
