using DevPilotAI.Api.Configuration;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Infrastructure.Health;

public sealed class ChromaHealthCheck : IHealthCheck
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ChromaSettings _settings;

    public ChromaHealthCheck(IHttpClientFactory httpClientFactory, IOptions<ChromaSettings> settings)
    {
        _httpClientFactory = httpClientFactory;
        _settings = settings.Value;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken ct = default)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("chromadb");
            var response = await client.GetAsync($"http://{_settings.Host}:{_settings.Port}/api/v1/heartbeat", ct);
            return response.IsSuccessStatusCode
                ? HealthCheckResult.Healthy("ChromaDB is reachable")
                : HealthCheckResult.Degraded($"ChromaDB returned {response.StatusCode}");
        }
        catch (Exception ex)
        {
            return HealthCheckResult.Unhealthy("ChromaDB is unreachable", ex);
        }
    }
}
