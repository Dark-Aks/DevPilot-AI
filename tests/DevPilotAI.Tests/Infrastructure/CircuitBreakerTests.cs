using DevPilotAI.Api.Infrastructure.Resilience;
using FluentAssertions;
using Xunit;

namespace DevPilotAI.Tests.Infrastructure;

public class CircuitBreakerTests
{
    [Fact]
    public async Task Should_Transition_To_Open_After_Threshold()
    {
        var breaker = new CircuitBreaker<string>(2, TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAsync<InvalidOperationException>(() => breaker.ExecuteAsync(() => throw new InvalidOperationException()));
        await Assert.ThrowsAsync<InvalidOperationException>(() => breaker.ExecuteAsync(() => throw new InvalidOperationException()));

        breaker.State.Should().Be(CircuitState.Open);
    }

    [Fact]
    public async Task Should_Transition_To_HalfOpen_After_Recovery_Window()
    {
        var breaker = new CircuitBreaker<string>(1, TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAsync<InvalidOperationException>(() => breaker.ExecuteAsync(() => throw new InvalidOperationException()));
        await Task.Delay(60);

        breaker.State.Should().Be(CircuitState.HalfOpen);
    }

    [Fact]
    public async Task Should_Transition_Back_To_Closed_On_Probe_Success()
    {
        var breaker = new CircuitBreaker<string>(1, TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAsync<InvalidOperationException>(() => breaker.ExecuteAsync(() => throw new InvalidOperationException()));
        await Task.Delay(60);
        var result = await breaker.ExecuteAsync(() => Task.FromResult("ok"));

        result.Should().Be("ok");
        breaker.State.Should().Be(CircuitState.Closed);
    }
}
