using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Rag;

namespace DevPilotAI.Api.Services;

public sealed class QueryService : IQueryService
{
    private readonly IRetriever _retriever;

    public QueryService(IRetriever retriever)
    {
        _retriever = retriever;
    }

    public async Task<QueryResponse> SearchAsync(QueryRequest request, CancellationToken ct = default)
    {
        var topK = request.TopK ?? 15;
        var results = await _retriever.RetrieveAsync(request.Query, request.Repo, topK, request.FilterLanguage, ct);
        return new QueryResponse(request.Repo, request.Query, results);
    }
}
