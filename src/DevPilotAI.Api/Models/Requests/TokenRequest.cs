namespace DevPilotAI.Api.Models.Requests;

public sealed class TokenRequest
{
    public string ApiKey { get; set; } = string.Empty;
}

public sealed class RefreshTokenRequest
{
    public string RefreshToken { get; set; } = string.Empty;
}
