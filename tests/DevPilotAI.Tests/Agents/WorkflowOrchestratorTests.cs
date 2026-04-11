using DevPilotAI.Api.Agents;
using FluentAssertions;
using Xunit;

namespace DevPilotAI.Tests.Agents;

public class WorkflowOrchestratorTests
{
    [Fact]
    public void Should_Dispatch_Agents_For_Api_Changes()
    {
        var selected = WorkflowOrchestrator.DispatchAgents(new HashSet<ChangeType> { ChangeType.Api });

        selected.Should().Contain("code_understanding");
        selected.Should().Contain("test_generator");
        selected.Should().Contain("review");
        selected.Should().NotContain("documentation");
    }
}
