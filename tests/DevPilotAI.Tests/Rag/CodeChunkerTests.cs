using DevPilotAI.Api.Rag;
using FluentAssertions;
using Xunit;

namespace DevPilotAI.Tests.Rag;

public class CodeChunkerTests
{
    [Fact]
    public async Task Should_Chunk_CSharp_Using_Roslyn()
    {
        var chunker = new CodeChunker();
        var code = "public class A { public void X(){} public void Y(){} }";

        var chunks = await chunker.ChunkAsync("a.cs", code, "csharp", "repo", "commit");

        chunks.Should().NotBeEmpty();
        chunks.Any(x => x.ChunkType == "class" || x.ChunkType == "function").Should().BeTrue();
    }
}
