using DevPilotAI.Api.Agents;

namespace DevPilotAI.Api.Models.Responses;

public sealed record WorkflowResponse(string Repo, string Branch, IReadOnlyList<AgentResult> Results, string RequestId);
