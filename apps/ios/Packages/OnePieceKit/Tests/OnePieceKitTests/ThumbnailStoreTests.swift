import Foundation
import Testing
@testable import OnePieceKit

@Suite struct ThumbnailStoreTests {
    actor Fetcher {
        var calls = 0
        var failing = false
        func setFailing(_ value: Bool) { failing = value }
        func fetch(_ url: URL) throws -> Data {
            calls += 1
            if failing { throw URLError(.notConnectedToInternet) }
            return Data("jpeg:\(url.lastPathComponent)".utf8)
        }
    }

    func directory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "thumbs-\(UUID())", directoryHint: .isDirectory)
    }

    func entry(_ id: String, art: String? = "https://example.com/art.jpg") -> CatalogEntry {
        CatalogEntry(printingId: id, cardId: id, name: "N", set: "OP-01", kind: "base", rarity: "C", artUrl: art)
    }

    @Test func fetchesOnceThenServesFromDisk() async throws {
        let fetcher = Fetcher()
        let dir = directory()
        let store = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        let first = await store.data(for: entry("OP01-001"))
        let second = await store.data(for: entry("OP01-001"))
        #expect(first == Data("jpeg:art.jpg".utf8) && second == first)
        #expect(await fetcher.calls == 1)
        #expect(FileManager.default.fileExists(atPath: dir.appending(path: "OP01-001.jpg").path))
        // A new store over the same directory also hits the cache.
        let reopened = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        #expect(await reopened.data(for: entry("OP01-001")) == first)
        #expect(await fetcher.calls == 1)
    }

    @Test func failedFetchIsNotCachedAndRetries() async {
        let fetcher = Fetcher()
        await fetcher.setFailing(true)
        let dir = directory()
        let store = ThumbnailStore(directory: dir) { try await fetcher.fetch($0) }
        #expect(await store.data(for: entry("OP01-002")) == nil)
        #expect(!FileManager.default.fileExists(atPath: dir.appending(path: "OP01-002.jpg").path))
        await fetcher.setFailing(false)
        #expect(await store.data(for: entry("OP01-002")) != nil)
        #expect(await fetcher.calls == 2)
    }

    @Test func noArtURLNeverFetches() async {
        let fetcher = Fetcher()
        let store = ThumbnailStore(directory: directory()) { try await fetcher.fetch($0) }
        #expect(await store.data(for: entry("OP01-003", art: nil)) == nil)
        #expect(await fetcher.calls == 0)
    }
}
