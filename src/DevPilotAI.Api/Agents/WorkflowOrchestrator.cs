using DevPilotAI.Api.Infrastructure.Metrics;
using DevPilotAI.Api.Rag;

namespace DevPilotAI.Api.Agents;

public sealed class WorkflowOrchestrator
{
    private static readonly Dictionary<ChangeType, string[]> RoutingTable = new()
    {
        [ChangeType.Api] = ["code_understanding", "test_generator", "review"],
        [ChangeType.Logic] = ["code_understanding", "review"],
        [ChangeType.Ui] = ["test_generator", "review"],
        [ChangeType.Config] = ["documentation", "review"],
        [ChangeType.Schema] = ["documentation", "review", "code_understanding"],
        [ChangeType.Docs] = ["documentation"],
        [ChangeType.Test] = ["review"],
        [ChangeType.Unknown] = ["code_understanding", "test_generator", "documentation", "review"]
    };

    private readonly IEnumerable<IAgent> _agents;
    private readonly IRetriever _retriever;
    private readonly ILogger<WorkflowOrchestrator> _logger;

    public WorkflowOrchestrator(IEnumerable<IAgent> agents, IRetriever retriever, ILogger<WorkflowOrchestrator> logger)
    {
        _agents = agents;
        _retriever = retriever;
        _logger = logger;
    }

    public async Task<IReadOnlyList<AgentResult>> RunAsync(AgentState state, CancellationToken ct = default)
    {
        var changeTypes = ClassifyChanges(state.ChangedFiles);
        state.ChangeTypes.UnionWith(changeTypes);

        var context = await _retriever.RetrieveAsync(state.Diff, state.Repo, 15, null, ct);
        state.RagContext.AddRange(context.Select(x => x.Chunk));

        var selectedNames = DispatchAgents(changeTypes);
        state.AgentsToRun.Clear();
        state.AgentsToRun.AddRange(selectedNames);

        var selectedAgents = _agents.Where(a => selectedNames.Contains(a.Name, StringComparer.OrdinalIgnoreCase)).ToList();
        var tasks = selectedAgents.Select(agent => RunAgentSafelyAsync(agent, state, ct));
        var results = await Task.WhenAll(tasks);

        CollectMetrics(state.Metrics, results, context.Count);
        return results;
    }

    public static HashSet<ChangeType> ClassifyChanges(IEnumerable<ChangedFile> changedFiles)
    {
        var output = new HashSet<ChangeType>();
        foreach (var file in changedFiles)
        {
            var path = file.Path.ToLowerInvariant();
            if (path.Contains("controller") || path.Contains("api/")) output.Add(ChangeType.Api);
            else if (path.Contains("service") || path.Contains("core") || path.Contains("logic")) output.Add(ChangeType.Logic);
            else if (path.Contains("ui") || path.Contains("component") || path.EndsWith(".tsx") || path.EndsWith(".jsx")) output.Add(ChangeType.Ui);
            else if (path.Contains("config") || path.EndsWith(".json") || path.EndsWith(".yml")) output.Add(ChangeType.Config);
            else if (path.Contains("schema") || path.Contains("model")) output.Add(ChangeType.Schema);
            else if (path.Contains("readme") || path.EndsWith(".md")) output.Add(ChangeType.Docs);
            else if (path.Contains("test") || path.Contains("spec")) output.Add(ChangeType.Test);
            else output.Add(ChangeType.Unknown);
        }

        if (output.Count == 0)
        {
            output.Add(ChangeType.Unknown);
        }

        return output;
    }

    public static IReadOnlyCollection<string> DispatchAgents(HashSet<ChangeType> changeTypes)
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var type in changeTypes)
        {
            if (RoutingTable.TryGetValue(type, out var mapped))
            {
                foreach (var m in mapped)
                {
                    names.Add(m);
                }
            }
        }

        if (names.Count == 0)
        {
            foreach (var m in RoutingTable[ChangeType.Unknown])
            {
                names.Add(m);
            }
        }

        return names;
    }

    private async Task<AgentResult> RunAgentSafelyAsync(IAgent agent, AgentState state, CancellationToken ct)
    {
        try
        {
            return await agent.RunAsync(state, ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Agent {AgentName} failed unexpectedly", agent.Name);
            return new AgentResult(agent.Name, false, "Unhandled fallback", 0, 0, 0, 0m, "{}");
        }
    }

    private static void CollectMetrics(RequestMetrics metrics, IReadOnlyList<AgentResult> results, int retrievalCount)
    {
        metrics.AgentsInvoked = results.Count;
        metrics.RetrievalChunks = retrievalCount;
        metrics.TotalInputTokens = results.Sum(x => x.InputTokens);
        metrics.TotalOutputTokens = results.Sum(x => x.OutputTokens);
        metrics.TotalLatencyMs = results.Sum(x => x.LatencyMs);
        metrics.Errors = results.Count(x => !x.Success);
        metrics.EstimateCost();
    }
}
