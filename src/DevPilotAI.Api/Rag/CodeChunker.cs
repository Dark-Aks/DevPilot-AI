using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis.CSharp;

namespace DevPilotAI.Api.Rag;

public sealed class CodeChunker : IChunker
{
    public Task<IReadOnlyList<CodeChunk>> ChunkAsync(string filePath, string content, string language, string repoName, string commitId, CancellationToken ct = default)
    {
        if (language.Equals("c#", StringComparison.OrdinalIgnoreCase) || filePath.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult<IReadOnlyList<CodeChunk>>(ChunkCSharp(filePath, content, repoName, commitId));
        }

        return Task.FromResult<IReadOnlyList<CodeChunk>>(ChunkWithTreeSitterStyleFallback(filePath, content, language, repoName, commitId));
    }

    private static List<CodeChunk> ChunkCSharp(string filePath, string content, string repoName, string commitId)
    {
        var tree = CSharpSyntaxTree.ParseText(content);
        var root = tree.GetRoot();
        var chunks = new List<CodeChunk>();

        foreach (var node in root.DescendantNodes())
        {
            var type = node.GetType().Name;
            if (type is "ClassDeclarationSyntax" or "MethodDeclarationSyntax")
            {
                var span = tree.GetLineSpan(node.Span);
                var symbolName = node.ToString().Split('{', StringSplitOptions.TrimEntries)[0].Trim();

                chunks.Add(new CodeChunk(
                    Id: Guid.NewGuid().ToString("N"),
                    Content: node.ToFullString(),
                    Language: "csharp",
                    FilePath: filePath,
                    StartLine: span.StartLinePosition.Line + 1,
                    EndLine: span.EndLinePosition.Line + 1,
                    ChunkType: type.StartsWith("Class", StringComparison.OrdinalIgnoreCase) ? "class" : "function",
                    RepoName: repoName,
                    CommitId: commitId,
                    SymbolName: symbolName));
            }
        }

        if (chunks.Count == 0)
        {
            chunks.Add(new CodeChunk(Guid.NewGuid().ToString("N"), content, "csharp", filePath, 1, content.Split('\n').Length, "file", repoName, commitId, Path.GetFileNameWithoutExtension(filePath)));
        }

        return chunks;
    }

    private static List<CodeChunk> ChunkWithTreeSitterStyleFallback(string filePath, string content, string language, string repoName, string commitId)
    {
        // Process-based tree-sitter approach can be plugged in here; fallback uses boundary heuristics.
        var chunks = new List<CodeChunk>();
        var lines = content.Split('\n');
        var boundary = new Regex(@"^(def |class |function |export function |const .*=>)", RegexOptions.Compiled);
        var start = 0;

        for (var i = 0; i < lines.Length; i++)
        {
            if (i > start && boundary.IsMatch(lines[i]))
            {
                chunks.Add(BuildChunk(start, i - 1));
                start = i;
            }
        }

        chunks.Add(BuildChunk(start, lines.Length - 1));
        return chunks;

        CodeChunk BuildChunk(int s, int e)
        {
            var segment = string.Join("\n", lines[s..(e + 1)]);
            var chunkType = boundary.IsMatch(lines[s]) ? "function" : "file";
            return new CodeChunk(Guid.NewGuid().ToString("N"), segment, language.ToLowerInvariant(), filePath, s + 1, e + 1, chunkType, repoName, commitId, Path.GetFileNameWithoutExtension(filePath));
        }
    }
}
