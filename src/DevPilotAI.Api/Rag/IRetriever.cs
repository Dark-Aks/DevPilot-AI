namespace DevPilotAI.Api.Rag;

public interface IRetriever
{
    Task<IReadOnlyList<RetrievedChunk>> RetrieveAsync(string query, string repo, int topK, string? filterLanguage = null, CancellationToken ct = default);
}
