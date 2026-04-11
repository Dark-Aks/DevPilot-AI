using DevPilotAI.Api.Agents;
using DevPilotAI.Api.Rag;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using Xunit;

namespace DevPilotAI.Tests.Agents;

public class AgentFallbackTests
{
    [Fact]
    public async Task Should_Continue_When_One_Agent_Fails()
    {
        var failing = new Mock<IAgent>();
        failing.SetupGet(x => x.Name).Returns("review");
        failing.Setup(x => x.RunAsync(It.IsAny<AgentState>(), It.IsAny<CancellationToken>())).ThrowsAsync(new Exception("boom"));

        var passing = new Mock<IAgent>();
        passing.SetupGet(x => x.Name).Returns("documentation");
        passing.Setup(x => x.RunAsync(It.IsAny<AgentState>(), It.IsAny<CancellationToken>())).ReturnsAsync(new AgentResult("documentation", true, "ok", 0, 0, 1, 0m, "{}"));

        var retriever = new Mock<IRetriever>();
        retriever.Setup(x => x.RetrieveAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<string?>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<RetrievedChunk>());

        var orchestrator = new WorkflowOrchestrator(new[] { failing.Object, passing.Object }, retriever.Object, NullLogger<WorkflowOrchestrator>.Instance);
        var state = new AgentState { Repo = "x", Diff = "y", ChangedFiles = [new ChangedFile("README.md", "modified")] };

        var results = await orchestrator.RunAsync(state);
        results.Should().HaveCount(1); // only mapped docs agent is selected for docs changes
        results.Single().Agent.Should().Be("documentation");
    }
}
