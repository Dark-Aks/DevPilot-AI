namespace DevPilotAI.Api.Agents;

public interface IAgent
{
    string Name { get; }
    Task<AgentResult> RunAsync(AgentState state, CancellationToken ct = default);
}
