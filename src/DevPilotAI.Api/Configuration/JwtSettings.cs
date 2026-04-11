namespace DevPilotAI.Api.Configuration;

public sealed class JwtSettings
{
    public string Secret { get; set; } = string.Empty;
    public int ExpiryMinutes { get; set; } = 60;
    public string Issuer { get; set; } = "DevPilotAI";
    public string Audience { get; set; } = "DevPilotAI";
    public int RefreshExpiryMinutes { get; set; } = 1440;
}
