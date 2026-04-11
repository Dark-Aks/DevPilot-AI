using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace DevPilotAI.Api.Infrastructure.Health;

public sealed class OpenAIHealthCheck : IHealthCheck
{
    private readonly IHttpClientFactory _httpClientFactory;

    public OpenAIHealthCheck(IHttpClientFactory httpClientFactory) => _httpClientFactory = httpClientFactory;

    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken ct = default)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("openai");
            var baseUrl = Environment.GetEnvironmentVariable("OPENAI_BASE_URL") ?? "https://api.openai.com";
            var response = await client.GetAsync($"{baseUrl.TrimEnd('/')}/v1/models", ct);
            return response.IsSuccessStatusCode
                ? HealthCheckResult.Healthy("OpenAI API is reachable")
                : HealthCheckResult.Degraded($"OpenAI API returned {response.StatusCode}");
        }
        catch (Exception ex)
        {
            return HealthCheckResult.Unhealthy("OpenAI API is unreachable", ex);
        }
    }
}
