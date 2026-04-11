using System.Text;
using Asp.Versioning;
using DevPilotAI.Api.Agents;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Infrastructure.Health;
using DevPilotAI.Api.Infrastructure.Resilience;
using DevPilotAI.Api.Infrastructure.Metrics;
using DevPilotAI.Api.Middleware;
using DevPilotAI.Api.Rag;
using DevPilotAI.Api.Services;
using FluentValidation;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using Microsoft.SemanticKernel;
using Microsoft.OpenApi.Models;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using StackExchange.Redis;

namespace DevPilotAI.Api.Extensions;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddDevPilotServices(this IServiceCollection services, IConfiguration config, IWebHostEnvironment env)
    {
        services.AddControllers();
        services.AddEndpointsApiExplorer();

        // Configuration bindings
        services.Configure<LlmSettings>(config.GetSection("Llm"));
        services.Configure<EmbeddingSettings>(config.GetSection("Embedding"));
        services.Configure<GitHubSettings>(config.GetSection("GitHub"));
        services.Configure<RagSettings>(config.GetSection("Rag"));
        services.Configure<CacheSettings>(config.GetSection("Cache"));
        services.Configure<ChromaSettings>(config.GetSection("Chroma"));
        services.Configure<AppSettings>(config.GetSection("App"));
        services.Configure<JwtSettings>(config.GetSection("Jwt"));

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

        services.PostConfigure<JwtSettings>(x =>
        {
            x.Secret = Environment.GetEnvironmentVariable("JWT_SECRET") ?? x.Secret;
            if (int.TryParse(Environment.GetEnvironmentVariable("JWT_EXPIRY_MINUTES"), out var exp))
            {
                x.ExpiryMinutes = exp;
            }
            x.Issuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? x.Issuer;
            x.Audience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? x.Audience;
        });

        // API Versioning
        services.AddApiVersioning(options =>
        {
            options.DefaultApiVersion = new ApiVersion(1, 0);
            options.AssumeDefaultVersionWhenUnspecified = true;
            options.ReportApiVersions = true;
        }).AddApiExplorer(options =>
        {
            options.GroupNameFormat = "'v'VVV";
            options.SubstituteApiVersionInUrl = true;
        });

        // JWT Authentication
        var jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET")
            ?? config.GetValue<string>("Jwt:Secret")
            ?? "DevPilotAI-Default-Secret-Change-Me-In-Production-32chars!";

        services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret)),
                    ValidateIssuer = true,
                    ValidIssuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? config.GetValue<string>("Jwt:Issuer") ?? "DevPilotAI",
                    ValidateAudience = true,
                    ValidAudience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? config.GetValue<string>("Jwt:Audience") ?? "DevPilotAI",
                    ValidateLifetime = true,
                    ClockSkew = TimeSpan.FromMinutes(1)
                };
            });

        services.AddAuthorization();

        // Cache provider toggle
        services.AddMemoryCache();
        var cacheProvider = Environment.GetEnvironmentVariable("CACHE_PROVIDER") ?? "memory";
        if (cacheProvider.Equals("redis", StringComparison.OrdinalIgnoreCase))
        {
            var redisConn = config.GetValue<string>("Redis:ConnectionString") ?? "localhost:6379";
            services.AddSingleton<IConnectionMultiplexer>(_ => ConnectionMultiplexer.Connect(redisConn));
            services.AddSingleton<ICacheProvider, RedisCacheProvider>();
        }
        else
        {
            services.AddSingleton<ICacheProvider, MemoryCacheProvider>();
        }

        services.AddHttpClient("github");
        services.AddHttpClient("chromadb");
        services.AddHttpClient("openai");

        // Multi-origin CORS
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
                    var origins = (Environment.GetEnvironmentVariable("CORS_ORIGINS")
                        ?? config.GetValue<string>("CORS_ORIGINS")
                        ?? "https://example.com")
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

                    policy.WithOrigins(origins).AllowAnyHeader().AllowAnyMethod();
                }
            });
        });

        // Health checks
        var healthBuilder = services.AddHealthChecks()
            .AddCheck<ChromaHealthCheck>("chromadb", tags: ["dependency"])
            .AddCheck<OpenAIHealthCheck>("openai", tags: ["dependency"]);

        if (cacheProvider.Equals("redis", StringComparison.OrdinalIgnoreCase))
        {
            var redisConn = config.GetValue<string>("Redis:ConnectionString") ?? "localhost:6379";
            healthBuilder.AddRedis(redisConn, name: "redis", tags: ["dependency"]);
        }

        // Swagger
        services.AddSwaggerGen(c =>
        {
            c.SwaggerDoc("v1", new OpenApiInfo { Title = "DevPilot AI API", Version = "v1" });
            c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
            {
                Description = "JWT Bearer token. Example: \"Bearer {token}\"",
                Name = "Authorization",
                In = ParameterLocation.Header,
                Type = SecuritySchemeType.Http,
                Scheme = "bearer",
                BearerFormat = "JWT"
            });
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
                        Reference = new OpenApiReference { Id = "Bearer", Type = ReferenceType.SecurityScheme }
                    },
                    Array.Empty<string>()
                },
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference { Id = "ApiKey", Type = ReferenceType.SecurityScheme }
                    },
                    Array.Empty<string>()
                }
            });
        });

        // OpenTelemetry
        services.AddOpenTelemetry()
            .ConfigureResource(res => res.AddService("DevPilotAI"))
            .WithTracing(tracing =>
            {
                tracing
                    .AddAspNetCoreInstrumentation()
                    .AddHttpClientInstrumentation()
                    .AddSource("DevPilotAI.Agents")
                    .AddSource("DevPilotAI.Rag")
                    .AddSource("DevPilotAI.GitHub");

                if (env.IsDevelopment())
                {
                    tracing.AddConsoleExporter();
                }
                else
                {
                    tracing.AddOtlpExporter();
                }
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
