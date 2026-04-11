using System.Text.Json;
using StackExchange.Redis;

namespace DevPilotAI.Api.Infrastructure.Caching;

public sealed class RedisCacheProvider : ICacheProvider
{
    private readonly IConnectionMultiplexer _redis;

    public RedisCacheProvider(IConnectionMultiplexer redis) => _redis = redis;

    public async Task<T?> GetAsync<T>(string key, CancellationToken ct = default)
    {
        var db = _redis.GetDatabase();
        var value = await db.StringGetAsync(key);
        return value.HasValue ? JsonSerializer.Deserialize<T>(value!) : default;
    }

    public async Task SetAsync<T>(string key, T value, TimeSpan ttl, CancellationToken ct = default)
    {
        var db = _redis.GetDatabase();
        var json = JsonSerializer.Serialize(value);
        await db.StringSetAsync(key, json, ttl);
    }

    public async Task RemoveAsync(string key, CancellationToken ct = default)
    {
        var db = _redis.GetDatabase();
        await db.KeyDeleteAsync(key);
    }
}
