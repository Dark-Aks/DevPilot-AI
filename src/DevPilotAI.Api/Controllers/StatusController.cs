using Asp.Versioning;
using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Models.Responses;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[ApiVersion("1.0")]
[Route("api/v{version:apiVersion}/status")]
[Authorize]
public sealed class StatusController : ControllerBase
{
    private readonly ICacheProvider _cache;

    public StatusController(ICacheProvider cache) => _cache = cache;

    [HttpGet("{requestId}")]
    [ProducesResponseType(typeof(StatusResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetStatus(string requestId, CancellationToken ct)
    {
        var result = await _cache.GetAsync<object>($"workflow:{requestId}", ct);
        if (result is null)
        {
            return Ok(new StatusResponse(requestId, "pending"));
        }

        return Ok(new StatusResponse(requestId, "completed", result));
    }
}
