using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using Asp.Versioning;
using DevPilotAI.Api.Configuration;
using DevPilotAI.Api.Infrastructure.Caching;
using DevPilotAI.Api.Models.Requests;
using DevPilotAI.Api.Models.Responses;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;

namespace DevPilotAI.Api.Controllers;

[ApiController]
[ApiVersion("1.0")]
[Route("api/v{version:apiVersion}/auth")]
[AllowAnonymous]
public sealed class AuthController : ControllerBase
{
    private readonly AppSettings _appSettings;
    private readonly JwtSettings _jwtSettings;
    private readonly ICacheProvider _cache;

    public AuthController(IOptions<AppSettings> appSettings, IOptions<JwtSettings> jwtSettings, ICacheProvider cache)
    {
        _appSettings = appSettings.Value;
        _jwtSettings = jwtSettings.Value;
        _cache = cache;
    }

    [HttpPost("token")]
    [ProducesResponseType(typeof(TokenResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status401Unauthorized)]
    public async Task<IActionResult> Token([FromBody] TokenRequest request, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(_appSettings.ApiKey) || request.ApiKey != _appSettings.ApiKey)
        {
            return Unauthorized(new { error = "Invalid API key" });
        }

        var (accessToken, refreshToken) = GenerateTokens();
        await _cache.SetAsync($"refresh:{refreshToken}", true, TimeSpan.FromMinutes(_jwtSettings.RefreshExpiryMinutes), ct);

        return Ok(new TokenResponse(accessToken, refreshToken, _jwtSettings.ExpiryMinutes * 60));
    }

    [HttpPost("refresh")]
    [ProducesResponseType(typeof(TokenResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status401Unauthorized)]
    public async Task<IActionResult> Refresh([FromBody] RefreshTokenRequest request, CancellationToken ct)
    {
        var stored = await _cache.GetAsync<bool>($"refresh:{request.RefreshToken}", ct);
        if (!stored)
        {
            return Unauthorized(new { error = "Invalid or expired refresh token" });
        }

        await _cache.RemoveAsync($"refresh:{request.RefreshToken}", ct);

        var (accessToken, newRefresh) = GenerateTokens();
        await _cache.SetAsync($"refresh:{newRefresh}", true, TimeSpan.FromMinutes(_jwtSettings.RefreshExpiryMinutes), ct);

        return Ok(new TokenResponse(accessToken, newRefresh, _jwtSettings.ExpiryMinutes * 60));
    }

    private (string AccessToken, string RefreshToken) GenerateTokens()
    {
        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSettings.Secret));
        var credentials = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, "devpilot-client"),
            new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
        };

        var token = new JwtSecurityToken(
            issuer: _jwtSettings.Issuer,
            audience: _jwtSettings.Audience,
            claims: claims,
            expires: DateTime.UtcNow.AddMinutes(_jwtSettings.ExpiryMinutes),
            signingCredentials: credentials);

        var accessToken = new JwtSecurityTokenHandler().WriteToken(token);
        var refreshToken = Convert.ToBase64String(RandomNumberGenerator.GetBytes(64));

        return (accessToken, refreshToken);
    }
}
