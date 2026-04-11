FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/DevPilotAI.Api/DevPilotAI.Api.csproj src/DevPilotAI.Api/
RUN dotnet restore src/DevPilotAI.Api/DevPilotAI.Api.csproj
COPY src/DevPilotAI.Api/ src/DevPilotAI.Api/
RUN dotnet publish src/DevPilotAI.Api/DevPilotAI.Api.csproj -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080
ENTRYPOINT ["dotnet", "DevPilotAI.Api.dll"]
