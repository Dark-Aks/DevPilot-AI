namespace DevPilotAI.Api.Rag;

public interface IChunker
{
    Task<IReadOnlyList<CodeChunk>> ChunkAsync(string filePath, string content, string language, string repoName, string commitId, CancellationToken ct = default);
}

public sealed record CodeChunk(
    string Id,
    string Content,
    string Language,
    string FilePath,
    int StartLine,
    int EndLine,
    string ChunkType,
    string RepoName,
    string CommitId,
    string SymbolName);
