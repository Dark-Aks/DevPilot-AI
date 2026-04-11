namespace DevPilotAI.Api.Configuration;

public sealed class CacheSettings
{
    public int TtlSeconds { get; init; } = 300;
    public int MaxSize { get; init; } = 1000;
}
