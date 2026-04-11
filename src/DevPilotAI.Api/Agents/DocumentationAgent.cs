using System.Diagnostics;
using System.Text.Json;
using DevPilotAI.Api.Infrastructure.Resilience;
using Microsoft.SemanticKernel;

namespace DevPilotAI.Api.Agents;

[AgentFallback("Documentation generation failed; fallback output emitted")]
public sealed class DocumentationAgent : IAgent
{
    public string Name => "documentation";

    private readonly Kernel _kernel;
    private readonly CircuitBreaker<string> _breaker;

    public DocumentationAgent(Kernel kernel)
    {
        _kernel = kernel;
        _breaker = new CircuitBreaker<string>(5, TimeSpan.FromSeconds(60));
    }

    public async Task<AgentResult> RunAsync(AgentState state, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var prompt = $"Write docs update notes. Return JSON with summary and changelog. Diff:\n{state.Diff}";
            var response = await _breaker.ExecuteAsync(async () =>
            {
                var result = await _kernel.InvokePromptAsync(prompt, cancellationToken: ct);
                return result.ToString() ?? "{}";
            });

            sw.Stop();
            return new AgentResult(Name, true, "Documentation complete", 300, 200, sw.ElapsedMilliseconds, 0m, EnsureJson(response));
        }
        catch (Exception ex)
        {
            sw.Stop();
            return new AgentResult(Name, false, $"Fallback: {ex.Message}", 0, 0, sw.ElapsedMilliseconds, 0m, "{\"summary\":\"fallback\"}");
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
            return JsonSerializer.Serialize(new { summary = input });
        }
    }
}
