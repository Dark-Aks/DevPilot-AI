using DevPilotAI.Api.Models.Responses;

namespace DevPilotAI.Api.Services;

public interface IIngestionService
{
    Task<IngestResponse> IngestRepositoryAsync(string repoUrl, string branch, CancellationToken ct = default);
}
