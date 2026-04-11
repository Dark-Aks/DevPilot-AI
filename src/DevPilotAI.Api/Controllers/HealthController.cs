using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Models.Responses;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[Route("health")]
[AllowAnonymous]
public sealed class HealthController : ControllerBase
{
    private static readonly DateTimeOffset StartedAt = DateTimeOffset.UtcNow;
    private readonly IWebHostEnvironment _environment;
    private readonly CacheService _cacheService;
    private readonly HealthCheckService _healthCheckService;

    public HealthController(IWebHostEnvironment environment, CacheService cacheService, HealthCheckService healthCheckService)
    {
        _environment = environment;
        _cacheService = cacheService;
        _healthCheckService = healthCheckService;
    }

    [HttpGet]
    [ProducesResponseType(typeof(HealthResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(HealthResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> Get(CancellationToken ct)
    {
        var report = await _healthCheckService.CheckHealthAsync(ct);

        var dependencies = report.Entries.ToDictionary(
            e => e.Key,
            e => e.Value.Status.ToString());

        var response = new HealthResponse(
            Version: typeof(Program).Assembly.GetName().Version?.ToString() ?? "1.0.0",
            Environment: _environment.EnvironmentName,
            Uptime: DateTimeOffset.UtcNow - StartedAt,
            CacheHitRates: new Dictionary<string, double>
            {
                ["retrieval"] = _cacheService.GetHitRate("retrieval"),
                ["embedding"] = _cacheService.GetHitRate("embedding"),
                ["llm"] = _cacheService.GetHitRate("llm")
            },
            Status: report.Status.ToString(),
            Dependencies: dependencies);

        var statusCode = report.Status == HealthStatus.Healthy
            ? StatusCodes.Status200OK
            : StatusCodes.Status503ServiceUnavailable;

        return StatusCode(statusCode, response);
    }
}
