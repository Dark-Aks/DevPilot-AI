using System.Text;
using System.Text.Json;
using DevPilotAI.Api.Configuration;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Rag;

public sealed class ChromaVectorStore : IVectorStore
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ChromaSettings _settings;
    private readonly ILogger<ChromaVectorStore> _logger;

    public ChromaVectorStore(IHttpClientFactory httpClientFactory, IOptions<ChromaSettings> settings, ILogger<ChromaVectorStore> logger)
    {
        _httpClientFactory = httpClientFactory;
        _settings = settings.Value;
        _logger = logger;
    }

    public async Task UpsertAsync(string repo, IReadOnlyList<CodeChunk> chunks, IReadOnlyList<float[]> embeddings, CancellationToken ct = default)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("chromadb");
            var baseUrl = $"http://{_settings.Host}:{_settings.Port}";
            var collection = GetCollection(repo);

            var payload = new
            {
                ids = chunks.Select(c => c.Id).ToArray(),
                embeddings = embeddings.Select(e => e.Select(v => (double)v).ToArray()).ToArray(),
                documents = chunks.Select(c => c.Content).ToArray(),
                metadatas = chunks.Select(c => new Dictionary<string, object>
                {
                    ["language"] = c.Language,
                    ["filePath"] = c.FilePath,
                    ["startLine"] = c.StartLine,
                    ["endLine"] = c.EndLine,
                    ["chunkType"] = c.ChunkType,
                    ["repoName"] = c.RepoName,
                    ["commitId"] = c.CommitId,
                    ["symbolName"] = c.SymbolName
                }).ToArray()
            };

            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await client.PostAsync($"{baseUrl}/api/v1/collections/{collection}/upsert", content, ct);
            response.EnsureSuccessStatusCode();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "ChromaDB unavailable during upsert for repo {Repo}", repo);
        }
    }

    public async Task<IReadOnlyList<RetrievedChunk>> SimilaritySearchAsync(string repo, float[] queryEmbedding, int topK, string? filter = null, CancellationToken ct = default)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("chromadb");
            var baseUrl = $"http://{_settings.Host}:{_settings.Port}";
            var collection = GetCollection(repo);

            var payload = new
            {
                query_embeddings = new[] { queryEmbedding.Select(v => (double)v).ToArray() },
                n_results = topK,
                where = string.IsNullOrWhiteSpace(filter) ? null : new Dictionary<string, string> { ["language"] = filter }
            };

            var response = await client.PostAsync(
                $"{baseUrl}/api/v1/collections/{collection}/query",
                new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json"),
                ct);

            response.EnsureSuccessStatusCode();
            var body = await response.Content.ReadAsStringAsync(ct);
            return ParseResults(repo, body);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "ChromaDB unavailable during similarity search for repo {Repo}", repo);
            return [];
        }
    }

    public async Task<CollectionStats> GetCollectionStatsAsync(string repo, CancellationToken ct = default)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("chromadb");
            var baseUrl = $"http://{_settings.Host}:{_settings.Port}";
            var collection = GetCollection(repo);
            var response = await client.GetAsync($"{baseUrl}/api/v1/collections/{collection}", ct);
            response.EnsureSuccessStatusCode();
            var body = await response.Content.ReadAsStringAsync(ct);
            using var doc = JsonDocument.Parse(body);
            var count = doc.RootElement.TryGetProperty("count", out var c) ? c.GetInt32() : 0;
            return new CollectionStats(collection, count);
        }
        catch
        {
            return new CollectionStats(GetCollection(repo), 0);
        }
    }

    private static string GetCollection(string repo)
    {
        var sanitized = new string(repo.ToLowerInvariant().Select(ch => char.IsLetterOrDigit(ch) ? ch : '_').ToArray());
        return $"repo_{sanitized}";
    }

    private static IReadOnlyList<RetrievedChunk> ParseResults(string repo, string json)
    {
        using var doc = JsonDocument.Parse(json);
        var chunks = new List<RetrievedChunk>();

        if (!doc.RootElement.TryGetProperty("documents", out var documents) || documents.GetArrayLength() == 0)
        {
            return chunks;
        }

        var docs = documents[0];
        var distances = doc.RootElement.TryGetProperty("distances", out var dist) && dist.GetArrayLength() > 0 ? dist[0] : default;
        var metadatas = doc.RootElement.TryGetProperty("metadatas", out var meta) && meta.GetArrayLength() > 0 ? meta[0] : default;

        for (var i = 0; i < docs.GetArrayLength(); i++)
        {
            var docText = docs[i].GetString() ?? string.Empty;
            var metadata = metadatas.ValueKind == JsonValueKind.Array && i < metadatas.GetArrayLength() ? metadatas[i] : default;
            var chunk = new CodeChunk(
                Guid.NewGuid().ToString("N"),
                docText,
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("language", out var lang) ? lang.GetString() ?? "unknown" : "unknown",
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("filePath", out var path) ? path.GetString() ?? "unknown" : "unknown",
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("startLine", out var sl) ? sl.GetInt32() : 1,
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("endLine", out var el) ? el.GetInt32() : 1,
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("chunkType", out var ct) ? ct.GetString() ?? "file" : "file",
                repo,
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("commitId", out var cid) ? cid.GetString() ?? string.Empty : string.Empty,
                metadata.ValueKind == JsonValueKind.Object && metadata.TryGetProperty("symbolName", out var sym) ? sym.GetString() ?? string.Empty : string.Empty);

            var score = distances.ValueKind == JsonValueKind.Array && i < distances.GetArrayLength() ? 1d / (1d + distances[i].GetDouble()) : 0d;
            chunks.Add(new RetrievedChunk(chunk, score));
        }

        return chunks;
    }
}
