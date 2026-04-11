using FluentValidation;

namespace DevPilotAI.Api.Models.Requests;

public sealed record IngestRequest(string RepoUrl, string Branch = "main");

public sealed class IngestRequestValidator : AbstractValidator<IngestRequest>
{
    public IngestRequestValidator()
    {
        RuleFor(x => x.RepoUrl).NotEmpty().Must(x => Uri.TryCreate(x, UriKind.Absolute, out _));
        RuleFor(x => x.Branch).NotEmpty();
    }
}
