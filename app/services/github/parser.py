from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Node

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Language registry ──

LANGUAGES: dict[str, Language] = {}

_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

# Node types that represent top-level code constructs per language
_CAPTURE_NODES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "expression_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "expression_statement",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "expression_statement",
    },
}


def _init_languages() -> None:
    """Initialize tree-sitter Language objects once."""
    if LANGUAGES:
        return
    LANGUAGES["python"] = Language(tspython.language())
    LANGUAGES["javascript"] = Language(tsjavascript.language())
    LANGUAGES["typescript"] = Language(tstypescript.language_typescript())
    LANGUAGES["tsx"] = Language(tstypescript.language_tsx())


def detect_language(file_path: str) -> str | None:
    """Return the language key for a file path, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


# ── Data classes ──


@dataclass
class CodeChunk:
    name: str
    chunk_type: str  # function | class | method | module
    code: str
    start_line: int
    end_line: int
    language: str
    docstring: str = ""
    parent_class: str = ""
    decorators: list[str] = field(default_factory=list)


# ── Extraction helpers ──


def _get_node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_name(node: Node, source: bytes) -> str:
    """Extract the name identifier from a node."""
    for child in node.children:
        if child.type == "identifier":
            return _get_node_text(child, source)
        if child.type == "property_identifier":
            return _get_node_text(child, source)
        if child.type == "name":
            return _get_node_text(child, source)
    return "<anonymous>"


def _extract_docstring_python(node: Node, source: bytes) -> str:
    """Extract docstring from the first expression_statement child in a Python body."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for expr in stmt.children:
                        if expr.type == "string":
                            raw = _get_node_text(expr, source)
                            return raw.strip("\"'").strip()
                    break
            break
    return ""


def _extract_js_name_from_declaration(node: Node, source: bytes) -> str:
    """Extract name from a JS/TS variable declaration (const foo = ...)."""
    for child in node.children:
        if child.type == "variable_declarator":
            return _extract_name(child, source)
        if child.type == "variable_declaration":
            return _extract_js_name_from_declaration(child, source)
    return _extract_name(node, source)


def _classify_node(node: Node, language: str) -> str:
    """Map a tree-sitter node type to our chunk_type vocabulary."""
    t = node.type
    if "class" in t:
        return "class"
    if "function" in t or "method" in t:
        return "function"
    if "interface" in t:
        return "interface"
    if "type_alias" in t:
        return "type"
    return "statement"


def _extract_python_methods(class_node: Node, source: bytes, language: str) -> list[CodeChunk]:
    """Extract methods from a Python class body."""
    methods: list[CodeChunk] = []
    class_name = _extract_name(class_node, source)
    for child in class_node.children:
        if child.type == "block":
            for stmt in child.children:
                target = stmt
                decos: list[str] = []
                if stmt.type == "decorated_definition":
                    for d in stmt.children:
                        if d.type == "decorator":
                            decos.append(_get_node_text(d, source))
                        elif d.type == "function_definition":
                            target = d
                            break
                if target.type == "function_definition":
                    methods.append(
                        CodeChunk(
                            name=_extract_name(target, source),
                            chunk_type="method",
                            code=_get_node_text(stmt, source),
                            start_line=stmt.start_point[0] + 1,
                            end_line=stmt.end_point[0] + 1,
                            language=language,
                            docstring=_extract_docstring_python(target, source),
                            parent_class=class_name,
                            decorators=decos,
                        )
                    )
    return methods


def _extract_js_methods(class_node: Node, source: bytes, language: str) -> list[CodeChunk]:
    """Extract methods from a JS/TS class body."""
    methods: list[CodeChunk] = []
    class_name = _extract_name(class_node, source)
    for child in class_node.children:
        if child.type == "class_body":
            for member in child.children:
                if member.type in ("method_definition", "public_field_definition"):
                    methods.append(
                        CodeChunk(
                            name=_extract_name(member, source),
                            chunk_type="method",
                            code=_get_node_text(member, source),
                            start_line=member.start_point[0] + 1,
                            end_line=member.end_point[0] + 1,
                            language=language,
                            parent_class=class_name,
                        )
                    )
    return methods


# ── Main parse function ──


def parse_file(content: str, language: str) -> list[CodeChunk]:
    """Parse source code into a list of CodeChunks using tree-sitter.

    Args:
        content: Source code as a string.
        language: One of 'python', 'javascript', 'typescript', 'tsx'.

    Returns:
        List of CodeChunk objects extracted from the file.
    """
    _init_languages()

    if language not in LANGUAGES:
        # Fallback: return the entire file as a single chunk
        lines = content.splitlines()
        return [
            CodeChunk(
                name="<module>",
                chunk_type="module",
                code=content,
                start_line=1,
                end_line=len(lines),
                language=language,
            )
        ]

    parser = Parser(LANGUAGES[language])
    source = content.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    capture_types = _CAPTURE_NODES.get(language, set())
    chunks: list[CodeChunk] = []

    for node in root.children:
        if node.type not in capture_types:
            continue

        actual_node = node
        decorators: list[str] = []

        # Handle decorated definitions (Python)
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    decorators.append(_get_node_text(child, source))
                elif child.type in ("function_definition", "class_definition"):
                    actual_node = child
                    break

        # Handle export statements (JS/TS)
        if node.type == "export_statement":
            for child in node.children:
                if child.type in (
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                    "interface_declaration",
                    "type_alias_declaration",
                ):
                    actual_node = child
                    break

        chunk_type = _classify_node(actual_node, language)
        if chunk_type == "statement" and actual_node.type == "lexical_declaration":
            name = _extract_js_name_from_declaration(actual_node, source)
        else:
            name = _extract_name(actual_node, source)

        docstring = ""
        if language == "python" and actual_node.type in (
            "function_definition",
            "class_definition",
        ):
            docstring = _extract_docstring_python(actual_node, source)

        chunk = CodeChunk(
            name=name,
            chunk_type=chunk_type,
            code=_get_node_text(node, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=language,
            docstring=docstring,
            decorators=decorators,
        )
        chunks.append(chunk)

        # Extract methods from classes
        if chunk_type == "class":
            if language == "python":
                chunks.extend(_extract_python_methods(actual_node, source, language))
            else:
                chunks.extend(_extract_js_methods(actual_node, source, language))

    # If no top-level constructs found, return entire file
    if not chunks:
        lines = content.splitlines()
        chunks.append(
            CodeChunk(
                name="<module>",
                chunk_type="module",
                code=content,
                start_line=1,
                end_line=len(lines),
                language=language,
            )
        )

    return chunks
