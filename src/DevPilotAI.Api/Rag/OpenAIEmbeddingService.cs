using System.Text;
using System.Text.Json;
using DevPilotAI.Api.Configuration;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Rag;

public sealed class OpenAIEmbeddingService : IEmbeddingService
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly EmbeddingSettings _settings;
    private readonly ILogger<OpenAIEmbeddingService> _logger;

    public OpenAIEmbeddingService(IHttpClientFactory httpClientFactory, IOptions<EmbeddingSettings> settings, ILogger<OpenAIEmbeddingService> logger)
    {
        _httpClientFactory = httpClientFactory;
        _settings = settings.Value;
        _logger = logger;
    }

    public async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var output = new List<float[]>();
        foreach (var batch in Batch(texts, _settings.BatchSize))
        {
            var embeddings = await EmbedBatchWithRetryAsync(batch, ct);
            output.AddRange(embeddings);
        }

        return output.ToArray();
    }

    public async Task<float[]> EmbedQueryAsync(string query, CancellationToken ct = default)
    {
        var result = await EmbedAsync([query], ct);
        return result[0];
    }

    private async Task<List<float[]>> EmbedBatchWithRetryAsync(IReadOnlyList<string> batch, CancellationToken ct)
    {
        var client = _httpClientFactory.CreateClient("openai");
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty;
        var endpoint = Environment.GetEnvironmentVariable("OPENAI_BASE_URL") ?? "https://api.openai.com/v1/embeddings";

        for (var attempt = 0; attempt < 5; attempt++)
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
            request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {apiKey}");
            request.Content = new StringContent(JsonSerializer.Serialize(new { model = _settings.Model, input = batch }), Encoding.UTF8, "application/json");

            var response = await client.SendAsync(request, ct);
            var body = await response.Content.ReadAsStringAsync(ct);

            if ((int)response.StatusCode == 429)
            {
                var delay = TimeSpan.FromMilliseconds(Math.Pow(2, attempt) * 300);
                _logger.LogWarning("OpenAI embedding rate-limited. Retry in {Delay} ms", delay.TotalMilliseconds);
                await Task.Delay(delay, ct);
                continue;
            }

            response.EnsureSuccessStatusCode();
            using var doc = JsonDocument.Parse(body);
            var data = doc.RootElement.GetProperty("data");
            var vectors = new List<float[]>();
            foreach (var item in data.EnumerateArray())
            {
                var embedding = item.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                vectors.Add(embedding);
            }

            return vectors;
        }

        throw new InvalidOperationException("Failed to generate embeddings after retries.");
    }

    private static IEnumerable<IReadOnlyList<string>> Batch(IReadOnlyList<string> values, int size)
    {
        for (var i = 0; i < values.Count; i += size)
        {
            yield return values.Skip(i).Take(size).ToList();
        }
    }
}
