namespace DevPilotAI.Api.Models.Responses;

public sealed record HealthResponse(string Version, string Environment, TimeSpan Uptime, Dictionary<string, double> CacheHitRates);
