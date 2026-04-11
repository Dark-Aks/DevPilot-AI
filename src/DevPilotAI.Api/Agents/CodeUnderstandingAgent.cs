using System.Diagnostics;
using System.Text.Json;
using DevPilotAI.Api.Infrastructure.Resilience;
using Microsoft.SemanticKernel;

namespace DevPilotAI.Api.Agents;

[AgentFallback("Code understanding failed; fallback summary emitted")]
public sealed class CodeUnderstandingAgent : IAgent
{
    public string Name => "code_understanding";

    private readonly Kernel _kernel;
    private readonly CircuitBreaker<string> _breaker;

    public CodeUnderstandingAgent(Kernel kernel)
    {
        _kernel = kernel;
        _breaker = new CircuitBreaker<string>(5, TimeSpan.FromSeconds(60));
    }

    public async Task<AgentResult> RunAsync(AgentState state, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var prompt = $"Summarize code intent. Return JSON with fields summary, risks, recommendations. Diff:\n{state.Diff}\nContext:\n{string.Join("\n---\n", state.RagContext.Select(x => x.Content).Take(3))}";
            var response = await _breaker.ExecuteAsync(async () =>
            {
                var result = await _kernel.InvokePromptAsync(prompt, cancellationToken: ct);
                return result.ToString() ?? "{}";
            });

            sw.Stop();
            return new AgentResult(Name, true, "Code understanding complete", 500, 200, sw.ElapsedMilliseconds, 0m, EnsureJson(response));
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
