using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DevPilotAI.Api.Agents;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Metrics;
using DevPilotAI.Api.Models.Requests;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[Route("api/webhook/github")]
[EnableRateLimiting("webhook")]
public sealed class WebhookController : ControllerBase
{
    private readonly GitHubSettings _githubSettings;
    private readonly WorkflowOrchestrator _orchestrator;
    private readonly RequestMetricsFactory _metricsFactory;
    private readonly ILogger<WebhookController> _logger;

    public WebhookController(
        IOptions<GitHubSettings> githubSettings,
        WorkflowOrchestrator orchestrator,
        RequestMetricsFactory metricsFactory,
        ILogger<WebhookController> logger)
    {
        _githubSettings = githubSettings.Value;
        _orchestrator = orchestrator;
        _metricsFactory = metricsFactory;
        _logger = logger;
    }

    [HttpPost]
    public async Task<IActionResult> Handle(CancellationToken ct)
    {
        using var reader = new StreamReader(Request.Body);
        var payloadRaw = await reader.ReadToEndAsync(ct);

        if (!Request.Headers.TryGetValue("X-Hub-Signature-256", out var signature) || !IsValidSignature(payloadRaw, signature.ToString()))
        {
            return Unauthorized(new { error = "Invalid webhook signature" });
        }

        var payload = JsonSerializer.Deserialize<WebhookPayload>(payloadRaw, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        if (payload is null)
        {
            return BadRequest(new { error = "Invalid payload" });
        }

        _ = Task.Run(async () =>
        {
            try
            {
                var changedFiles = payload.Commits
                    .SelectMany(c => c.Added.Concat(c.Modified).Concat(c.Removed))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Select(x => new ChangedFile(x, "modified"))
                    .ToList();

                var state = new AgentState
                {
                    Repo = payload.Repository.Full_Name,
                    Branch = payload.Ref,
                    ChangedFiles = changedFiles,
                    ChangeTypes = new HashSet<ChangeType>(),
                    AgentsToRun = new List<string>(),
                    RagContext = new List<Rag.CodeChunk>(),
                    Diff = string.Join("\n", payload.Commits.Select(c => c.Message)),
                    Metrics = _metricsFactory.Create()
                };

                await _orchestrator.RunAsync(state, CancellationToken.None);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Background webhook workflow failed");
            }
        }, CancellationToken.None);

        return Accepted(new { status = "workflow_queued" });
    }

    private bool IsValidSignature(string payload, string signatureHeader)
    {
        if (string.IsNullOrWhiteSpace(_githubSettings.WebhookSecret) || !signatureHeader.StartsWith("sha256=", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var expected = signatureHeader[7..];
        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(_githubSettings.WebhookSecret));
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(payload));
        var actual = Convert.ToHexString(hash).ToLowerInvariant();
        return CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(expected), Encoding.UTF8.GetBytes(actual));
    }
}
