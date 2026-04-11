using DevPilotAI.Api.Configuration;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Infrastructure.Caching;

public sealed class CacheService
{
    private readonly IMemoryCache _cache;
    private readonly TimeSpan _ttl;
    private readonly Dictionary<string, long> _hits = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, long> _misses = new(StringComparer.OrdinalIgnoreCase);

    public CacheService(IMemoryCache cache, IOptions<CacheSettings> settings)
    {
        _cache = cache;
        _ttl = TimeSpan.FromSeconds(settings.Value.TtlSeconds);
    }

    public bool TryGet<T>(string tier, string key, out T? value)
    {
        if (_cache.TryGetValue(key, out var obj) && obj is T typed)
        {
            _hits[tier] = _hits.GetValueOrDefault(tier) + 1;
            value = typed;
            return true;
        }

        _misses[tier] = _misses.GetValueOrDefault(tier) + 1;
        value = default;
        return false;
    }

    public void Set<T>(string tier, string key, T value)
    {
        _cache.Set(key, value, new MemoryCacheEntryOptions
        {
            SlidingExpiration = _ttl,
            Size = 1
        });
    }

    public double GetHitRate(string tier)
    {
        var hits = _hits.GetValueOrDefault(tier);
        var misses = _misses.GetValueOrDefault(tier);
        var total = hits + misses;
        return total == 0 ? 1.0 : hits / (double)total;
    }
}
