using DevPilotAI.Api.Middleware;
using Serilog;

namespace DevPilotAI.Api.Extensions;

public static class WebApplicationExtensions
{
    public static WebApplication UseDevPilotPipeline(this WebApplication app)
    {
        app.UseSerilogRequestLogging();
        app.UseCors("devpilot");

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseMiddleware<RequestLoggingMiddleware>();

        return app;
    }
}
