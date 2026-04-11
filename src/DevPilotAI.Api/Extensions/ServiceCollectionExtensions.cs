using DevPilotAI.Api.Agents;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Infrastructure.Resilience;
using DevPilotAI.Api.Infrastructure.Metrics;
using DevPilotAI.Api.Middleware;
using DevPilotAI.Api.Rag;
using DevPilotAI.Api.Services;
using FluentValidation;
using Microsoft.SemanticKernel;
using Microsoft.OpenApi.Models;

namespace DevPilotAI.Api.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddDevPilotServices(this IServiceCollection services, IConfiguration config, IWebHostEnvironment env)
    {
        services.AddControllers();
        services.AddEndpointsApiExplorer();

        services.Configure<LlmSettings>(config.GetSection("Llm"));
        services.Configure<EmbeddingSettings>(config.GetSection("Embedding"));
        services.Configure<GitHubSettings>(config.GetSection("GitHub"));
        services.Configure<RagSettings>(config.GetSection("Rag"));
        services.Configure<CacheSettings>(config.GetSection("Cache"));
        services.Configure<ChromaSettings>(config.GetSection("Chroma"));
        services.Configure<AppSettings>(config.GetSection("App"));

        services.PostConfigure<AppSettings>(x =>
        {
            var envApiKey = Environment.GetEnvironmentVariable("DEVPILOT_API_KEY");
            if (!string.IsNullOrWhiteSpace(envApiKey))
            {
                x.ApiKey = envApiKey;
            }
        });

        services.PostConfigure<GitHubSettings>(x =>
        {
            x.Token = Environment.GetEnvironmentVariable("GITHUB_TOKEN") ?? x.Token;
            x.WebhookSecret = Environment.GetEnvironmentVariable("GITHUB_WEBHOOK_SECRET") ?? x.WebhookSecret;
        });

        services.PostConfigure<ChromaSettings>(x =>
        {
            x.Host = Environment.GetEnvironmentVariable("CHROMA_HOST") ?? x.Host;
            if (int.TryParse(Environment.GetEnvironmentVariable("CHROMA_PORT"), out var port))
            {
                x.Port = port;
            }
        });

        services.AddMemoryCache();
        services.AddHttpClient("github");
        services.AddHttpClient("chromadb");
        services.AddHttpClient("openai");

        services.AddCors(options =>
        {
            options.AddPolicy("devpilot", policy =>
            {
                if (env.IsDevelopment())
                {
                    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod();
                }
                else
                {
                    policy.WithOrigins(config.GetValue<string>("CORS_ORIGIN") ?? "https://example.com")
                        .AllowAnyHeader()
                        .AllowAnyMethod();
                }
            });
        });

        services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo { Title = "DevPilot AI API", Version = "v1" });
            c.AddSecurityDefinition("ApiKey", new OpenApiSecurityScheme
            {
                Description = "API key needed to access endpoints. X-API-Key: {key}",
                Name = "X-API-Key",
                In = ParameterLocation.Header,
                Type = SecuritySchemeType.ApiKey
            });
            c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference { Id = "ApiKey", Type = ReferenceType.SecurityScheme }
                    },
                    Array.Empty<string>()
                }
            });
        });

        services.AddValidatorsFromAssemblyContaining<Program>();

        services.AddSingleton<RequestLoggingMiddleware>();
        services.AddScoped<ApiKeyMiddleware>();
        services.AddSingleton<CacheService>();
        services.AddSingleton<RequestMetricsFactory>();

        services.AddSingleton<IChunker, CodeChunker>();
        services.AddSingleton<IEmbeddingService, OpenAIEmbeddingService>();
        services.AddSingleton<IVectorStore, ChromaVectorStore>();
        services.AddSingleton<IRetriever, HybridRetriever>();

        services.AddScoped<IGitHubService, GitHubService>();
        services.AddScoped<IIngestionService, IngestionService>();
        services.AddScoped<IQueryService, QueryService>();

        services.AddSingleton<Kernel>(_ => Kernel.CreateBuilder().Build());
        services.AddScoped<IAgent, CodeUnderstandingAgent>();
        services.AddScoped<IAgent, TestGeneratorAgent>();
        services.AddScoped<IAgent, DocumentationAgent>();
        services.AddScoped<IAgent, ReviewAgent>();
        services.AddScoped<WorkflowOrchestrator>();

        return services;
    }
}
