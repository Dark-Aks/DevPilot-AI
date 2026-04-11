namespace DevPilotAI.Api.Rag;

public interface IEmbeddingService
{
    Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    Task<float[]> EmbedQueryAsync(string query, CancellationToken ct = default);
}
