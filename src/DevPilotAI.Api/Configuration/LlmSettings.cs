namespace DevPilotAI.Api.Configuration;

public sealed class LlmSettings
{
    public string Provider { get; init; } = "openai";
    public string Model { get; init; } = "gpt-4o";
    public double Temperature { get; init; } = 0.1;
    public int TimeoutSeconds { get; init; } = 120;
}
