namespace DevPilotAI.Api.Services;

public interface IGitHubService
{
    Task<string> GetRepositoryArchiveAsync(string repoUrl, string branch, CancellationToken ct = default);
    Task PostPullRequestCommentAsync(string repo, int prNumber, string body, CancellationToken ct = default);
}
