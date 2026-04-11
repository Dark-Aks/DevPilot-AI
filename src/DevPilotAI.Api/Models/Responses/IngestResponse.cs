namespace DevPilotAI.Api.Models.Responses;

public sealed record IngestResponse(string Repo, int FilesProcessed, int ChunksCreated, long DurationMs);
