using System.Text.Json;
using DevPilotAI.Api.Configuration;
using Microsoft.Extensions.Options;

namespace DevPilotAI.Api.Middleware;

public sealed class ApiKeyMiddleware : IMiddleware
{
    private static readonly HashSet<string> ExcludedPaths =
    [
        "/health",
        "/swagger",
        "/swagger/index.html",
        "/swagger/v1/swagger.json",
        "/api/v1/auth/token",
        "/api/v1/auth/refresh"
    ];

    private readonly ILogger<ApiKeyMiddleware> _logger;
    private readonly AppSettings _settings;

    public ApiKeyMiddleware(ILogger<ApiKeyMiddleware> logger, IOptions<AppSettings> options)
    {
        _logger = logger;
        _settings = options.Value;
    }

    public async Task InvokeAsync(HttpContext context, RequestDelegate next)
    {
        var path = context.Request.Path.Value ?? string.Empty;
        if (ExcludedPaths.Any(path.StartsWith))
        {
            await next(context);
            return;
        }

        if (!context.Request.Headers.TryGetValue("X-API-Key", out var apiKey) || string.IsNullOrWhiteSpace(_settings.ApiKey) || apiKey != _settings.ApiKey)
        {
            _logger.LogWarning("Unauthorized request from IP {IpAddress} on path {Path}", context.Connection.RemoteIpAddress?.ToString() ?? "unknown", path);
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            context.Response.ContentType = "application/json";
            await context.Response.WriteAsync(JsonSerializer.Serialize(new { error = "Invalid or missing API key" }));
            return;
        }

        await next(context);
    }
}
