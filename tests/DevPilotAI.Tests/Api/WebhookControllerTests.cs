using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Tests.Helpers;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using System.Net.Http.Headers;
using Xunit;

namespace DevPilotAI.Tests.Api;

public class WebhookControllerTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public WebhookControllerTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                services.Configure<GitHubSettings>(x => x.WebhookSecret = "test-secret");
                services.Configure<AppSettings>(x => x.ApiKey = "test-api-key");
                TestJwtHelper.ConfigureTestAuth(services);
            });
        });
    }

    [Fact]
    public async Task Should_Return_202_For_Valid_Hmac()
    {
        var client = _factory.CreateClient();
        var jwt = TestJwtHelper.GenerateToken();
        var payload = JsonSerializer.Serialize(new
        {
            @ref = "refs/heads/main",
            repository = new { name = "repo", full_name = "owner/repo", clone_url = "https://github.com/owner/repo" },
            commits = new[] { new { id = "1", message = "update", added = new[] { "a.cs" }, modified = Array.Empty<string>(), removed = Array.Empty<string>() } },
            pusher = new { name = "u", email = "e" }
        });

        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes("test-secret"));
        var sig = Convert.ToHexString(hmac.ComputeHash(Encoding.UTF8.GetBytes(payload))).ToLowerInvariant();

        var request = new HttpRequestMessage(HttpMethod.Post, "/api/v1/webhook/github");
        request.Headers.Add("X-Hub-Signature-256", $"sha256={sig}");
        request.Headers.Add("X-API-Key", "test-api-key");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", jwt);
        request.Content = new StringContent(payload, Encoding.UTF8, "application/json");

        var response = await client.SendAsync(request);
        response.StatusCode.Should().Be(System.Net.HttpStatusCode.Accepted);
    }
}
