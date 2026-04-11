namespace DevPilotAI.Api.Models.Responses;

public sealed record StatusResponse(string RequestId, string Status, object? Result = null);
