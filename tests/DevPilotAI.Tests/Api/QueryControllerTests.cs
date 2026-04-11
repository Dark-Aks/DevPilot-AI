using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using DevPilotAI.Api.Rag;
using DevPilotAI.Api.Services;
using DevPilotAI.Tests.Helpers;
using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using System.Net;
using System.Net.Http.Headers;
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
                TestJwtHelper.ConfigureTestAuth(services);
            });
        });

        var client = factory.CreateClient();
        var jwt = TestJwtHelper.GenerateToken();

        var req = new HttpRequestMessage(HttpMethod.Post, "/api/v1/query");
        req.Headers.Add("X-API-Key", "test-api-key");
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", jwt);
        req.Content = new StringContent("{\"query\":\"find auth\",\"repo\":\"repo\"}", Encoding.UTF8, "application/json");

        var res = await client.SendAsync(req);
        res.StatusCode.Should().Be(HttpStatusCode.OK);
    }
}
