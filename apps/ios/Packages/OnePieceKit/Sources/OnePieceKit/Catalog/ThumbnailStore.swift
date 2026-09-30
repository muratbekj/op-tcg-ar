import Foundation

/// Card art for catalog printings, fetched on demand from each printing's `artUrl` and cached on
/// disk as `<printingId>.jpg`. Only lists use these; recognition never touches the network.
/// Offline (or on any fetch error) `data(for:)` returns nil and caches nothing, so a later call can
/// still succeed.
public actor ThumbnailStore {
    public typealias Fetch = @Sendable (URL) async throws -> Data

    /// Downloads with URLSession, treating non-2xx responses as failures.
    public static let download: Fetch = { url in
        let (data, response) = try await URLSession.shared.data(from: url)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        return data
    }

    private let directory: URL
    private let fetch: Fetch

    public init(directory: URL, fetch: @escaping Fetch = ThumbnailStore.download) {
        self.directory = directory
        self.fetch = fetch
    }

    public func data(for entry: CatalogEntry) async -> Data? {
        let file = directory.appending(path: "\(entry.printingId).jpg")
        if let cached = try? Data(contentsOf: file) { return cached }
        guard let string = entry.artUrl, let url = URL(string: string),
              let data = try? await fetch(url), !data.isEmpty else { return nil }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? data.write(to: file, options: .atomic)
        return data
    }
}
