using DevPilotAI.Api.Infrastructure.Metrics;
using DevPilotAI.Api.Rag;

namespace DevPilotAI.Api.Agents;

public enum ChangeType
{
    Api,
    Logic,
    Ui,
    Config,
    Schema,
    Docs,
    Test,
    Unknown
}

public sealed record ChangedFile(string Path, string Status);

public sealed record AgentState
{
    public string Repo { get; init; } = string.Empty;
    public string Branch { get; init; } = "main";
    public List<ChangedFile> ChangedFiles { get; init; } = [];
    public HashSet<ChangeType> ChangeTypes { get; init; } = [];
    public List<string> AgentsToRun { get; init; } = [];
    public List<CodeChunk> RagContext { get; init; } = [];
    public string Diff { get; init; } = string.Empty;
    public RequestMetrics Metrics { get; init; } = new();
}

public sealed record AgentResult(
    string Agent,
    bool Success,
    string Summary,
    int InputTokens,
    int OutputTokens,
    long LatencyMs,
    decimal CostUsd,
    string RawJson);
