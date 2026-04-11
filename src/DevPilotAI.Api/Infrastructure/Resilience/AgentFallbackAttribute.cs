namespace DevPilotAI.Api.Infrastructure.Resilience;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
public sealed class AgentFallbackAttribute : Attribute
{
    public string Reason { get; }

    public AgentFallbackAttribute(string reason = "Agent fallback active")
    {
        Reason = reason;
    }
}
