using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using FluentAssertions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Xunit;

namespace DevPilotAI.Tests.Infrastructure;

public class CacheServiceTests
{
    [Fact]
    public void Should_Track_Hit_And_Miss_Per_Tier()
    {
        var service = new CacheService(new MemoryCache(new MemoryCacheOptions { SizeLimit = 1000 }), Options.Create(new CacheSettings()));

        service.TryGet<string>("retrieval", "missing", out _).Should().BeFalse();
        service.Set("retrieval", "k", "v");
        service.TryGet<string>("retrieval", "k", out var value).Should().BeTrue();
        value.Should().Be("v");

        service.GetHitRate("retrieval").Should().BeApproximately(0.5, 0.001);
    }
}
