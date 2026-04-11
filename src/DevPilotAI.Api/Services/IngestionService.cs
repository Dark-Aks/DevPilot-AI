using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Rag;

namespace DevPilotAI.Api.Services;

public sealed class IngestionService : IIngestionService
{
    private readonly IChunker _chunker;
    private readonly IEmbeddingService _embeddingService;
    private readonly IVectorStore _vectorStore;

    public IngestionService(IChunker chunker, IEmbeddingService embeddingService, IVectorStore vectorStore)
    {
        _chunker = chunker;
        _embeddingService = embeddingService;
        _vectorStore = vectorStore;
    }

    public async Task<IngestResponse> IngestRepositoryAsync(string repoUrl, string branch, CancellationToken ct = default)
    {
        var started = DateTimeOffset.UtcNow;
        var repo = repoUrl.Split('/').LastOrDefault() ?? "repo";

        // Placeholder ingestion over sample files. Replace with zip clone/extract scan in production.
        var sampleFiles = new List<(string path, string content, string language)>
        {
            ("sample.cs", "public class Sample { public void Run() {} }", "csharp")
        };

        var allChunks = new List<CodeChunk>();
        foreach (var file in sampleFiles)
        {
            var chunks = await _chunker.ChunkAsync(file.path, file.content, file.language, repo, Guid.NewGuid().ToString("N"), ct);
            allChunks.AddRange(chunks);
        }

        var embeddings = await _embeddingService.EmbedAsync(allChunks.Select(c => c.Content).ToList(), ct);
        await _vectorStore.UpsertAsync(repo, allChunks, embeddings, ct);

        return new IngestResponse(repo, sampleFiles.Count, allChunks.Count, (long)(DateTimeOffset.UtcNow - started).TotalMilliseconds);
    }
}
