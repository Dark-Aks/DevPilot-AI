using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[Route("api/query")]
[EnableRateLimiting("query")]
public sealed class QueryController : ControllerBase
{
    private readonly IQueryService _queryService;

    public QueryController(IQueryService queryService)
    {
        _queryService = queryService;
    }

    [HttpPost]
    [ProducesResponseType(typeof(QueryResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Query([FromBody] QueryRequest request, CancellationToken ct)
    {
        var result = await _queryService.SearchAsync(request, ct);
        return Ok(result);
    }
}
