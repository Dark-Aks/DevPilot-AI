namespace DevPilotAI.Api.Infrastructure.Metrics;

public sealed class RequestMetrics
{
    public string RequestId { get; set; } = Guid.NewGuid().ToString("N");
    public int TotalInputTokens { get; set; }
    public int TotalOutputTokens { get; set; }
    public decimal TotalCostUsd { get; set; }
    public long TotalLatencyMs { get; set; }
    public int RetrievalChunks { get; set; }
    public double RetrievalHitRate { get; set; }
    public int AgentsInvoked { get; set; }
    public int Errors { get; set; }

    public decimal EstimateCost()
    {
        var inputCost = (decimal)TotalInputTokens / 1000m * 0.0025m;
        var outputCost = (decimal)TotalOutputTokens / 1000m * 0.01m;
        TotalCostUsd = inputCost + outputCost;
        return TotalCostUsd;
    }
}

public sealed class RequestMetricsFactory
{
    public RequestMetrics Create() => new();
}
