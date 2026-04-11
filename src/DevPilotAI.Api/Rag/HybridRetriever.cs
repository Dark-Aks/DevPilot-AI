using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Rag;

public sealed class HybridRetriever : IRetriever
{
    private readonly IEmbeddingService _embeddingService;
    private readonly IVectorStore _vectorStore;
    private readonly CacheService _cache;
    private readonly RagSettings _settings;

    public HybridRetriever(IEmbeddingService embeddingService, IVectorStore vectorStore, CacheService cache, IOptions<RagSettings> settings)
    {
        _embeddingService = embeddingService;
        _vectorStore = vectorStore;
        _cache = cache;
        _settings = settings.Value;
    }

    public async Task<IReadOnlyList<RetrievedChunk>> RetrieveAsync(string query, string repo, int topK, string? filterLanguage = null, CancellationToken ct = default)
    {
        var cacheKey = $"retrieval:{repo}:{query}:{topK}:{filterLanguage}";
        if (_cache.TryGet<IReadOnlyList<RetrievedChunk>>("retrieval", cacheKey, out var cached) && cached is not null)
        {
            return cached;
        }

        var queryEmbedding = await _embeddingService.EmbedQueryAsync(query, ct);
        var candidates = await _vectorStore.SimilaritySearchAsync(repo, queryEmbedding, topK * 3, filterLanguage, ct);
        var reranked = ReRank(query, candidates, _settings.HybridAlpha)
            .Take(_settings.RerankTopK)
            .ToList();

        _cache.Set("retrieval", cacheKey, reranked);
        return reranked;
    }

    private static IEnumerable<RetrievedChunk> ReRank(string query, IReadOnlyList<RetrievedChunk> candidates, double alpha)
    {
        var queryTerms = Tokenize(query).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var docs = candidates.Select(x => x.Chunk.Content).ToList();
        var bm25Scores = ComputeBm25(queryTerms, docs);

        var ranked = candidates
            .Select((chunk, idx) =>
            {
                var vectorRank = chunk.VectorScore;
                var bm25 = bm25Scores[idx];
                var bonus = queryTerms.Any(t => chunk.Chunk.SymbolName.Contains(t, StringComparison.OrdinalIgnoreCase)) ? 0.3 : 0.0;
                var score = (alpha * vectorRank) + ((1 - alpha) * bm25) + bonus;
                return chunk with { FinalScore = score };
            })
            .OrderByDescending(x => x.FinalScore);

        return ranked;
    }

    private static double[] ComputeBm25(HashSet<string> queryTerms, List<string> docs)
    {
        const double k1 = 1.5;
        const double b = 0.75;
        var tokenized = docs.Select(Tokenize).ToList();
        var avgdl = tokenized.Average(x => x.Count);
        var n = docs.Count;

        var df = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var term in queryTerms)
        {
            df[term] = tokenized.Count(d => d.Contains(term, StringComparer.OrdinalIgnoreCase));
        }

        var scores = new double[n];
        for (var i = 0; i < n; i++)
        {
            var doc = tokenized[i];
            var tf = doc.GroupBy(x => x, StringComparer.OrdinalIgnoreCase).ToDictionary(g => g.Key, g => g.Count(), StringComparer.OrdinalIgnoreCase);

            foreach (var term in queryTerms)
            {
                if (!tf.TryGetValue(term, out var freq) || freq == 0)
                {
                    continue;
                }

                var idf = Math.Log(1 + (n - df[term] + 0.5) / (df[term] + 0.5));
                var numerator = freq * (k1 + 1);
                var denominator = freq + k1 * (1 - b + b * (doc.Count / avgdl));
                scores[i] += idf * (numerator / denominator);
            }
        }

        var max = scores.Max();
        if (max > 0)
        {
            for (var i = 0; i < scores.Length; i++)
            {
                scores[i] /= max;
            }
        }

        return scores;
    }

    private static List<string> Tokenize(string text)
    {
        return text.ToLowerInvariant()
            .Split([' ', '\n', '\r', '\t', '.', ',', ';', ':', '(', ')', '{', '}', '[', ']', '"', '\'', '/', '\\', '-', '_'], StringSplitOptions.RemoveEmptyEntries)
            .ToList();
    }
}
