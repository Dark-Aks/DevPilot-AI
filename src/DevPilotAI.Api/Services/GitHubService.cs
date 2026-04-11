using System.Text;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Resilience;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Services;

public sealed class GitHubService : IGitHubService
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly GitHubSettings _settings;
    private readonly CircuitBreaker<string> _breaker;

    public GitHubService(IHttpClientFactory httpClientFactory, IOptions<GitHubSettings> settings)
    {
        _httpClientFactory = httpClientFactory;
        _settings = settings.Value;
        _breaker = new CircuitBreaker<string>(_settings.CircuitBreakerThreshold, TimeSpan.FromSeconds(_settings.CircuitBreakerRecoverySeconds));
    }

    public Task<string> GetRepositoryArchiveAsync(string repoUrl, string branch, CancellationToken ct = default)
    {
        return _breaker.ExecuteAsync(async () =>
        {
            var client = _httpClientFactory.CreateClient("github");
            client.DefaultRequestHeaders.UserAgent.ParseAdd("DevPilotAI/1.0");
            if (!string.IsNullOrWhiteSpace(_settings.Token))
            {
                client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _settings.Token);
            }

            var url = repoUrl.TrimEnd('/') + $"/archive/refs/heads/{branch}.zip";
            return await client.GetStringAsync(url, ct);
        });
    }

    public Task PostPullRequestCommentAsync(string repo, int prNumber, string body, CancellationToken ct = default)
    {
        return _breaker.ExecuteAsync(async () =>
        {
            var client = _httpClientFactory.CreateClient("github");
            client.DefaultRequestHeaders.UserAgent.ParseAdd("DevPilotAI/1.0");
            client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _settings.Token);
            var url = $"https://api.github.com/repos/{repo}/issues/{prNumber}/comments";
            var response = await client.PostAsync(url, new StringContent($"{{\"body\":{System.Text.Json.JsonSerializer.Serialize(body)}}}", Encoding.UTF8, "application/json"), ct);
            response.EnsureSuccessStatusCode();
            return "ok";
        });
    }
}
