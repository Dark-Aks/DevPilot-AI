using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[Route("api/ingest")]
[EnableRateLimiting("ingest")]
public sealed class IngestController : ControllerBase
{
    private readonly IIngestionService _ingestionService;

    public IngestController(IIngestionService ingestionService)
    {
        _ingestionService = ingestionService;
    }

    [HttpPost]
    [ProducesResponseType(typeof(IngestResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Ingest([FromBody] IngestRequest request, CancellationToken ct)
    {
        var result = await _ingestionService.IngestRepositoryAsync(request.RepoUrl, request.Branch, ct);
        return Ok(result);
    }
}
