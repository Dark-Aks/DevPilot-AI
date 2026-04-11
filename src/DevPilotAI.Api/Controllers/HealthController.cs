using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Models.Responses;
using Microsoft.AspNetCore.Mvc;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[Route("health")]
public sealed class HealthController : ControllerBase
{
    private static readonly DateTimeOffset StartedAt = DateTimeOffset.UtcNow;
    private readonly IWebHostEnvironment _environment;
    private readonly CacheService _cacheService;

    public HealthController(IWebHostEnvironment environment, CacheService cacheService)
    {
        _environment = environment;
        _cacheService = cacheService;
    }

    [HttpGet]
    [ProducesResponseType(typeof(HealthResponse), StatusCodes.Status200OK)]
    public IActionResult Get()
    {
        var response = new HealthResponse(
            Version: typeof(Program).Assembly.GetName().Version?.ToString() ?? "1.0.0",
            Environment: _environment.EnvironmentName,
            Uptime: DateTimeOffset.UtcNow - StartedAt,
            CacheHitRates: new Dictionary<string, double>
            {
                ["retrieval"] = _cacheService.GetHitRate("retrieval"),
                ["embedding"] = _cacheService.GetHitRate("embedding"),
                ["llm"] = _cacheService.GetHitRate("llm")
            });

        return Ok(response);
    }
}
