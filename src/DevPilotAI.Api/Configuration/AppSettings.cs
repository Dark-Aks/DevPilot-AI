namespace DevPilotAI.Api.Configuration;

public sealed class AppSettings
{
    public string ApiKey { get; set; } = string.Empty;
    public string Environment { get; set; } = "development";
    public string LogLevel { get; set; } = "Information";
}

public sealed class EmbeddingSettings
{
    public string Model { get; set; } = "text-embedding-3-small";
    public int BatchSize { get; set; } = 100;
}

public sealed class ChromaSettings
{
    public string PersistDir { get; set; } = "./data/chroma";
    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 8001;
}
