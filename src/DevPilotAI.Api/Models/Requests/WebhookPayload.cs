namespace DevPilotAI.Api.Models.Requests;

public sealed record WebhookPayload(
    string Ref,
    WebhookRepository Repository,
    List<WebhookCommit> Commits,
    WebhookPusher Pusher);

public sealed record WebhookRepository(string Name, string Full_Name, string Clone_Url);
public sealed record WebhookCommit(string Id, string Message, List<string> Added, List<string> Modified, List<string> Removed);
public sealed record WebhookPusher(string Name, string Email);
