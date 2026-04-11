# Contributing to DevPilot AI

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- [Docker](https://www.docker.com/get-started) (for integration tests and local ChromaDB)
- Git

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Dark-Aks/DevPilot-AI.git
   cd DevPilot-AI
   ```

2. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```

3. Fill in required secrets in `.env` (at minimum `DEVPILOT_API_KEY` and `JWT_SECRET`).

4. Start ChromaDB:
   ```bash
   docker compose -f docker/docker-compose.yml up chromadb -d
   ```

5. Run the API:
   ```bash
   dotnet run --project src/DevPilotAI.Api
   ```

## Running Tests

All tests:
```bash
dotnet test DevPilotAI.sln
```

Unit tests only:
```bash
dotnet test DevPilotAI.sln --filter "FullyQualifiedName!~Integration"
```

Integration tests (requires Docker):
```bash
dotnet test DevPilotAI.sln --filter "FullyQualifiedName~Integration"
```

With coverage:
```bash
dotnet test DevPilotAI.sln --collect:"XPlat Code Coverage"
```

## Code Style

- Follow standard C# conventions and project patterns.
- Use `sealed` on classes that are not designed for inheritance.
- Prefer interfaces for service abstractions (`IService` → `Service`).
- Use records for immutable DTOs.
- Keep controllers thin — delegate logic to services.
- All new features should include unit tests.

## PR Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Keep PRs focused — one feature or fix per PR.
3. Ensure `dotnet build` passes with zero warnings and zero errors.
4. Ensure all tests pass (`dotnet test`).
5. Coverage must not drop below 80%.
6. Update documentation if your change affects the API or configuration.
7. Fill in the PR template with a clear description.

## Conventional Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `test:` | Adding or updating tests |
| `chore:` | Build process, CI, or dependency changes |
| `perf:` | Performance improvement |

Examples:
```
feat: add JWT refresh token endpoint
fix: correct HMAC validation for empty secrets
docs: update API reference with versioned routes
test: add auth controller integration tests
chore: bump OpenTelemetry packages to 1.10
```
