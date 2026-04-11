using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Rag;
using DevPilotAI.Api.Services;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using System.Net;
using System.Text;
using Xunit;

namespace DevPilotAI.Tests.Api;

public class QueryControllerTests : IClassFixture<WebApplicationFactory<Program>>
{
    [Fact]
    public async Task Should_Return_Ok_For_Query()
    {
        var query = new Mock<IQueryService>();
        query.Setup(x => x.SearchAsync(It.IsAny<QueryRequest>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new QueryResponse("repo", "q", new List<RetrievedChunk>()));

        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                services.AddSingleton(query.Object);
                services.Configure<AppSettings>(x => x.ApiKey = "test-api-key");
            });
        });

        var client = factory.CreateClient();

        var req = new HttpRequestMessage(HttpMethod.Post, "/api/query");
        req.Headers.Add("X-API-Key", "test-api-key");
        req.Content = new StringContent("{\"query\":\"find auth\",\"repo\":\"repo\"}", Encoding.UTF8, "application/json");

        var res = await client.SendAsync(req);
        res.StatusCode.Should().Be(HttpStatusCode.OK);
    }
}
