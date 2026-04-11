namespace DevPilotAI.Api.Configuration;

public sealed class RagSettings
{
    public int TopK { get; init; } = 15;
    public int RerankTopK { get; init; } = 8;
    public double HybridAlpha { get; init; } = 0.7;
}
