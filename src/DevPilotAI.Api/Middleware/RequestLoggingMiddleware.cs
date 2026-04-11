namespace DevPilotAI.Api.Middleware;

public sealed class RequestLoggingMiddleware : IMiddleware
{
    private readonly ILogger<RequestLoggingMiddleware> _logger;

    public RequestLoggingMiddleware(ILogger<RequestLoggingMiddleware> logger)
    {
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, RequestDelegate next)
    {
        var started = DateTimeOffset.UtcNow;
        await next(context);
        var elapsed = DateTimeOffset.UtcNow - started;

        _logger.LogInformation(
            "Request completed {Method} {Path} => {StatusCode} in {DurationMs}ms",
            context.Request.Method,
            context.Request.Path,
            context.Response.StatusCode,
            elapsed.TotalMilliseconds);
    }
}
