using DevPilotAI.Api.Rag;

namespace DevPilotAI.Api.Models.Responses;

public sealed record QueryResponse(string Repo, string Query, IReadOnlyList<RetrievedChunk> Results);
