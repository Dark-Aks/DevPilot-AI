namespace DevPilotAI.Api.Models.Responses;

public sealed record TokenResponse(string AccessToken, string RefreshToken, int ExpiresInSeconds);
