using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;

namespace DevPilotAI.Api.Services;

public interface IQueryService
{
    Task<QueryResponse> SearchAsync(QueryRequest request, CancellationToken ct = default);
}
