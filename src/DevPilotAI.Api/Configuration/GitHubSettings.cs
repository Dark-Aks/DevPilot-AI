namespace DevPilotAI.Api.Configuration;

public sealed class GitHubSettings
{
    public string Token { get; set; } = string.Empty;
    public string WebhookSecret { get; set; } = string.Empty;
    public int CircuitBreakerThreshold { get; set; } = 5;
    public int CircuitBreakerRecoverySeconds { get; set; } = 60;
}
