using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Services;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using System.Net;
using System.Text;
using Xunit;

namespace DevPilotAI.Tests.Api;

public class IngestControllerTests : IClassFixture<WebApplicationFactory<Program>>
{
    [Fact]
    public async Task Should_Apply_Rate_Limit_After_10_Requests()
    {
        var ingestion = new Mock<IIngestionService>();
        ingestion.Setup(x => x.IngestRepositoryAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new IngestResponse("repo", 1, 1, 10));

        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                services.AddSingleton(ingestion.Object);
                services.Configure<AppSettings>(x => x.ApiKey = "test-api-key");
            });
        });

        var client = factory.CreateClient();

        HttpStatusCode? lastStatus = null;
        for (var i = 0; i < 11; i++)
        {
            var req = new HttpRequestMessage(HttpMethod.Post, "/api/ingest");
            req.Headers.Add("X-API-Key", "test-api-key");
            req.Content = new StringContent("{\"repoUrl\":\"https://github.com/o/r\",\"branch\":\"main\"}", Encoding.UTF8, "application/json");
            var res = await client.SendAsync(req);
            lastStatus = res.StatusCode;
        }

        lastStatus.Should().Be(HttpStatusCode.TooManyRequests);
    }
}
