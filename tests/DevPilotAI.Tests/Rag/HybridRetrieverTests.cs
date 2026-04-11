using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Rag;
using FluentAssertions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Moq;
using Xunit;

namespace DevPilotAI.Tests.Rag;

public class HybridRetrieverTests
{
    [Fact]
    public async Task Should_Return_Reranked_Results()
    {
        var embedding = new Mock<IEmbeddingService>();
        embedding.Setup(x => x.EmbedQueryAsync("find login", It.IsAny<CancellationToken>())).ReturnsAsync([0.1f, 0.2f]);

        var store = new Mock<IVectorStore>();
        store.Setup(x => x.SimilaritySearchAsync("repo", It.IsAny<float[]>(), 9, null, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<RetrievedChunk>
            {
                new(new CodeChunk("1", "public void Login() {}", "csharp", "a.cs", 1, 3, "function", "repo", "c1", "Login"), 0.6),
                new(new CodeChunk("2", "public void Logout() {}", "csharp", "b.cs", 1, 3, "function", "repo", "c1", "Logout"), 0.9)
            });

        var cache = new CacheService(new MemoryCache(new MemoryCacheOptions { SizeLimit = 1000 }), Options.Create(new CacheSettings()));
        var retriever = new HybridRetriever(embedding.Object, store.Object, cache, Options.Create(new RagSettings { RerankTopK = 2, HybridAlpha = 0.7 }));

        var results = await retriever.RetrieveAsync("find login", "repo", 3);
        results.Should().NotBeEmpty();
        results.First().Chunk.SymbolName.Should().Be("Login");
    }
}
