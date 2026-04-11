namespace DevPilotAI.Api.Rag;

public interface IVectorStore
{
    Task UpsertAsync(string repo, IReadOnlyList<CodeChunk> chunks, IReadOnlyList<float[]> embeddings, CancellationToken ct = default);
    Task<IReadOnlyList<RetrievedChunk>> SimilaritySearchAsync(string repo, float[] queryEmbedding, int topK, string? filter = null, CancellationToken ct = default);
    Task<CollectionStats> GetCollectionStatsAsync(string repo, CancellationToken ct = default);
}

public sealed record RetrievedChunk(CodeChunk Chunk, double VectorScore, double FinalScore = 0d);
public sealed record CollectionStats(string Collection, int Count);
