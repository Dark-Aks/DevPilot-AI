using FluentValidation;

namespace DevPilotAI.Api.Models.Requests;

public sealed record QueryRequest(string Query, string Repo, int? TopK = null, string? FilterLanguage = null);

public sealed class QueryRequestValidator : AbstractValidator<QueryRequest>
{
    public QueryRequestValidator()
    {
        RuleFor(x => x.Query).NotEmpty().MinimumLength(3);
        RuleFor(x => x.Repo).NotEmpty();
        RuleFor(x => x.TopK).InclusiveBetween(1, 100).When(x => x.TopK.HasValue);
    }
}
