using System.Diagnostics;
using System.Text.Json;
using DevPilotAI.Api.Infrastructure.Resilience;
using Microsoft.SemanticKernel;

namespace DevPilotAI.Api.Agents;

[AgentFallback("Review failed; fallback output emitted")]
public sealed class ReviewAgent : IAgent
{
    public string Name => "review";

    private readonly Kernel _kernel;
    private readonly CircuitBreaker<string> _breaker;

    public ReviewAgent(Kernel kernel)
    {
        _kernel = kernel;
        _breaker = new CircuitBreaker<string>(5, TimeSpan.FromSeconds(60));
    }

    public async Task<AgentResult> RunAsync(AgentState state, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var prompt = $"Review this diff for issues. Return JSON with issues and suggestions. Diff:\n{state.Diff}";
            var response = await _breaker.ExecuteAsync(async () =>
            {
                var result = await _kernel.InvokePromptAsync(prompt, cancellationToken: ct);
                return result.ToString() ?? "{}";
            });

            sw.Stop();
            return new AgentResult(Name, true, "Review complete", 450, 250, sw.ElapsedMilliseconds, 0m, EnsureJson(response));
        }
        catch (Exception ex)
        {
            sw.Stop();
            return new AgentResult(Name, false, $"Fallback: {ex.Message}", 0, 0, sw.ElapsedMilliseconds, 0m, "{\"issues\":[]}");
        }
    }

    private static string EnsureJson(string input)
    {
        try
        {
            _ = JsonDocument.Parse(input);
            return input;
        }
        catch
        {
            return JsonSerializer.Serialize(new { issues = new[] { input } });
        }
    }
}
