namespace DevPilotAI.Api.Infrastructure.Resilience;

public enum CircuitState
{
    Closed,
    Open,
    HalfOpen
}

public sealed class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message)
    {
    }
}

public sealed class CircuitBreaker<T>
{
    private readonly int _threshold;
    private readonly TimeSpan _recoveryTime;
    private int _consecutiveFailures;
    private DateTimeOffset? _openedAt;
    private bool _halfOpenProbeUsed;

    public CircuitBreaker(int threshold, TimeSpan recoveryTime)
    {
        _threshold = threshold;
        _recoveryTime = recoveryTime;
    }

    public CircuitState State
    {
        get
        {
            if (_openedAt is null)
            {
                return CircuitState.Closed;
            }

            if (DateTimeOffset.UtcNow - _openedAt >= _recoveryTime)
            {
                return CircuitState.HalfOpen;
            }

            return CircuitState.Open;
        }
    }

    public async Task<T> ExecuteAsync(Func<Task<T>> operation)
    {
        var currentState = State;
        if (currentState == CircuitState.Open)
        {
            throw new CircuitBreakerOpenException("Circuit breaker is open.");
        }

        if (currentState == CircuitState.HalfOpen)
        {
            if (_halfOpenProbeUsed)
            {
                throw new CircuitBreakerOpenException("Circuit breaker is half-open and probe already used.");
            }

            _halfOpenProbeUsed = true;
        }

        try
        {
            var result = await operation();
            _consecutiveFailures = 0;
            _openedAt = null;
            _halfOpenProbeUsed = false;
            return result;
        }
        catch
        {
            _consecutiveFailures++;
            if (currentState == CircuitState.HalfOpen || _consecutiveFailures >= _threshold)
            {
                _openedAt = DateTimeOffset.UtcNow;
            }

            throw;
        }
    }
}
