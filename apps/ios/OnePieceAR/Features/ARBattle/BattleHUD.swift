import BattleKit
import SwiftUI

struct BattleHUD: View {
    @Environment(AppModel.self) private var model

    private var battle: BattleCoordinator { model.battle }

    var body: some View {
        VStack(spacing: 12) {
            if let engine = battle.engine {
                HStack(alignment: .top) {
                    FighterPanel(label: "P1", fighter: engine.left, isTurn: engine.turn == .left && !engine.isOver)
                    Spacer()
                    FighterPanel(label: "P2", fighter: engine.right, isTurn: engine.turn == .right && !engine.isOver)
                }
            }
            if let event = battle.lastEvent {
                Text(event).font(.headline).padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            controls.controlSize(.large)
        }
    }

    @ViewBuilder private var controls: some View {
        switch battle.phase {
        case .waiting, .resolving:
            EmptyView()
        case .ready:
            Button {
                battle.beginAttack()
            } label: {
                Text("\(battle.engine?.turn == .left ? "P1" : "P2") Attack").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
        case .charging:
            Button {
                battle.tapDon()
            } label: {
                VStack(spacing: 2) {
                    Text("DON!!").font(.title.weight(.black))
                    Text("Tap for +1000 (\(battle.donTaps)/\(BattleRules.maxDonTaps))").font(.caption)
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
        case .over:
            HStack {
                Button("Rematch") { battle.rematch() }.buttonStyle(.borderedProminent)
                Button("New cards") { model.clearAll() }.buttonStyle(.bordered)
            }
        }
    }
}

private struct FighterPanel: View {
    let label: String
    let fighter: Fighter
    let isTurn: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(label) · \(fighter.name)").font(.caption.weight(.semibold)).lineLimit(1)
            HStack(spacing: 3) {
                ForEach(0..<fighter.maxLife, id: \.self) { index in
                    Capsule()
                        .fill(index < fighter.life ? Color.green : Color.gray.opacity(0.4))
                        .frame(width: 16, height: 8)
                }
            }
            Text("\(fighter.power) power · \(fighter.countersLeft) counters").font(.caption2).foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            if isTurn { RoundedRectangle(cornerRadius: 12).strokeBorder(.orange, lineWidth: 2) }
        }
    }
}
