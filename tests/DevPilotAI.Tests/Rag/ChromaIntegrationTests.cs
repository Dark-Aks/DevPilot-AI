using System.Net.Http.Json;
using DotNet.Testcontainers.Builders;
using DotNet.Testcontainers.Containers;
using FluentAssertions;
using Xunit;

namespace DevPilotAI.Tests.Rag;

public class ChromaIntegrationTests : IAsyncLifetime
{
    private readonly IContainer _container = new ContainerBuilder()
        .WithImage("chromadb/chroma:latest")
        .WithPortBinding(8001, 8000)
        .WithWaitStrategy(Wait.ForUnixContainer().UntilPortIsAvailable(8000))
        .Build();

    public async Task InitializeAsync() => await _container.StartAsync();
    public async Task DisposeAsync() => await _container.DisposeAsync();

    [Fact(Skip = "Requires Docker runtime and Chroma API setup in CI environment")]
    public async Task Should_Upsert_And_Query()
    {
        using var client = new HttpClient { BaseAddress = new Uri("http://localhost:8001") };
        var heartbeat = await client.GetAsync("/api/v1/heartbeat");
        heartbeat.IsSuccessStatusCode.Should().BeTrue();

        var payload = new
        {
            ids = Enumerable.Range(1, 10).Select(i => i.ToString()).ToArray(),
            documents = Enumerable.Range(1, 10).Select(i => i == 1 ? "authenticate user login" : $"sample document {i}").ToArray(),
            embeddings = Enumerable.Range(1, 10).Select(_ => new[] { 0.01, 0.02, 0.03 }).ToArray(),
            metadatas = Enumerable.Range(1, 10).Select(i => new { language = "csharp", filePath = $"f{i}.cs" }).ToArray()
        };

        var response = await client.PostAsJsonAsync("/api/v1/collections/repo_test/upsert", payload);
        response.IsSuccessStatusCode.Should().BeTrue();
    }
}
