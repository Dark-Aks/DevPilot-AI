using Microsoft.Extensions.Caching.Memory;

namespace DevPilotAI.Api.Infrastructure.Caching;

public sealed class MemoryCacheProvider : ICacheProvider
{
    private readonly IMemoryCache _cache;

    public MemoryCacheProvider(IMemoryCache cache) => _cache = cache;

    public Task<T?> GetAsync<T>(string key, CancellationToken ct = default)
    {
        _cache.TryGetValue(key, out T? value);
        return Task.FromResult(value);
    }

    public Task SetAsync<T>(string key, T value, TimeSpan ttl, CancellationToken ct = default)
    {
        _cache.Set(key, value, new MemoryCacheEntryOptions { AbsoluteExpirationRelativeToNow = ttl });
        return Task.CompletedTask;
    }

    public Task RemoveAsync(string key, CancellationToken ct = default)
    {
        _cache.Remove(key);
        return Task.CompletedTask;
    }
}
