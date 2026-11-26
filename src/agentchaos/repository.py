"""Read-only, AST-based intelligence for local Python repositories."""

import ast
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import tokenize
from typing import Literal


DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)


class RepositoryAnalysisError(ValueError):
    """Base error for invalid repository inspection operations."""


class InvalidRepositoryPath(RepositoryAnalysisError):
    """A repository or source path is missing, unsafe, or unavailable."""


class OversizedSourceError(RepositoryAnalysisError):
    """A source file exceeds the configured read limit."""


@dataclass(frozen=True, slots=True)
class SourceLocation:
    path: str
    line: int
    column: int
    end_line: int


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    name: str
    qualified_name: str
    kind: Literal["function", "async_function", "class"]
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class ImportInfo:
    module: str
    name: str | None
    alias: str | None
    level: int
    scope: str | None
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class CallCandidate:
    """A syntactic call candidate, not a resolved runtime relationship."""

    callee: str
    caller: str | None
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class DecoratorInfo:
    symbol: str
    expression: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class AnalysisIssue:
    path: str
    kind: Literal[
        "syntax_error",
        "oversized",
        "read_error",
        "unsafe_symlink",
        "symlink_directory_skipped",
    ]
    message: str
    line: int | None = None


def _location(path: str, node: ast.AST) -> SourceLocation:
    line = getattr(node, "lineno", 1)
    return SourceLocation(
        path=path,
        line=line,
        column=getattr(node, "col_offset", 0) + 1,
        end_line=getattr(node, "end_lineno", line),
    )


def _expression(node: ast.AST) -> str:
    return ast.unparse(node)


class _Indexer(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.symbols: list[SymbolInfo] = []
        self.imports: list[ImportInfo] = []
        self.calls: list[CallCandidate] = []
        self.decorators: list[DecoratorInfo] = []

    def _qualified(self, name: str) -> str:
        return ".".join((*self.scope, name))

    def _scope(self) -> str | None:
        return ".".join(self.scope) or None

    def _symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        kind: Literal["function", "async_function", "class"],
    ) -> None:
        qualified_name = self._qualified(node.name)
        self.symbols.append(
            SymbolInfo(node.name, qualified_name, kind, _location(self.path, node))
        )
        self.decorators.extend(
            DecoratorInfo(
                symbol=qualified_name,
                expression=_expression(decorator),
                location=_location(self.path, decorator),
            )
            for decorator in node.decorator_list
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._symbol(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._symbol(node, "async_function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._symbol(node, "class")

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self.imports.append(
                ImportInfo(
                    module=imported.name,
                    name=None,
                    alias=imported.asname,
                    level=0,
                    scope=self._scope(),
                    location=_location(self.path, node),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            self.imports.append(
                ImportInfo(
                    module=node.module or "",
                    name=imported.name,
                    alias=imported.asname,
                    level=node.level,
                    scope=self._scope(),
                    location=_location(self.path, node),
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(
            CallCandidate(
                callee=_expression(node.func),
                caller=self._scope(),
                location=_location(self.path, node),
            )
        )
        self.generic_visit(node)


class RepositoryAnalyzer:
    """Build and query an isolated, deterministic AST index for one repository."""

    def __init__(
        self,
        root: str | Path,
        *,
        excluded_directories: set[str] | frozenset[str] | None = None,
        max_file_size: int = 1_000_000,
    ) -> None:
        requested_root = Path(root)
        try:
            resolved_root = requested_root.resolve(strict=True)
        except OSError as exception:
            raise InvalidRepositoryPath(f"repository does not exist: {root}") from exception
        if not resolved_root.is_dir():
            raise InvalidRepositoryPath(f"repository is not a directory: {root}")
        if max_file_size < 1:
            raise ValueError("max_file_size must be positive")

        self.root = resolved_root
        self.excluded_directories = DEFAULT_EXCLUDED_DIRECTORIES | frozenset(
            excluded_directories or ()
        )
        self.max_file_size = max_file_size
        self._files: list[str] = []
        self._sources: dict[str, str] = {}
        self._oversized: set[str] = set()
        self._symbols: list[SymbolInfo] = []
        self._imports: list[ImportInfo] = []
        self._calls: list[CallCandidate] = []
        self._decorators: list[DecoratorInfo] = []
        self._issues: list[AnalysisIssue] = []
        self._build_index()

    @property
    def issues(self) -> tuple[AnalysisIssue, ...]:
        return tuple(self._issues)

    def list_files(self) -> tuple[str, ...]:
        return tuple(self._files)

    def read_source(self, path: str | Path) -> str:
        relative = self._relative_path(path)
        if relative in self._oversized:
            raise OversizedSourceError(
                f"source exceeds {self.max_file_size} bytes: {relative}"
            )
        try:
            return self._sources[relative]
        except KeyError as exception:
            raise InvalidRepositoryPath(f"source is not readable: {relative}") from exception

    def find_symbol(self, name: str) -> tuple[SymbolInfo, ...]:
        return tuple(
            symbol
            for symbol in self._symbols
            if symbol.name == name or symbol.qualified_name == name
        )

    def get_imports(self, path: str | Path | None = None) -> tuple[ImportInfo, ...]:
        relative = self._relative_path(path) if path is not None else None
        return tuple(
            imported
            for imported in self._imports
            if relative is None or imported.location.path == relative
        )

    def find_callers(self, symbol: str) -> tuple[CallCandidate, ...]:
        qualified = "." in symbol
        return tuple(
            call
            for call in self._calls
            if call.callee == symbol
            or (not qualified and call.callee.rsplit(".", 1)[-1] == symbol)
        )

    def get_decorators(self, symbol: str | None = None) -> tuple[DecoratorInfo, ...]:
        return tuple(
            decorator
            for decorator in self._decorators
            if symbol is None
            or decorator.symbol == symbol
            or ("." not in symbol and decorator.symbol.rsplit(".", 1)[-1] == symbol)
        )

    def _relative_path(self, path: str | Path) -> str:
        raw = str(path)
        pure = PurePosixPath(raw.replace(os.sep, "/"))
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise InvalidRepositoryPath(f"path must be repository-relative: {path}")
        relative = pure.as_posix()
        if relative not in self._files:
            raise InvalidRepositoryPath(f"path is not an indexed Python file: {relative}")
        return relative

    def _build_index(self) -> None:
        for current, directories, files in os.walk(self.root, followlinks=False):
            current_path = Path(current)
            kept_directories: list[str] = []
            for directory in sorted(directories):
                if directory in self.excluded_directories:
                    continue
                candidate = current_path / directory
                if candidate.is_symlink():
                    relative = candidate.relative_to(self.root).as_posix()
                    kind = (
                        "unsafe_symlink"
                        if not self._safe_target(candidate)
                        else "symlink_directory_skipped"
                    )
                    self._issues.append(
                        AnalysisIssue(relative, kind, "symbolic-link directory skipped")
                    )
                    continue
                kept_directories.append(directory)
            directories[:] = kept_directories

            for filename in sorted(files):
                if not filename.endswith(".py"):
                    continue
                self._index_file(current_path / filename)

        self._files.sort()
        self._symbols.sort(key=lambda item: (item.location.path, item.location.line, item.name))
        self._imports.sort(
            key=lambda item: (item.location.path, item.location.line, item.module, item.name or "")
        )
        self._calls.sort(
            key=lambda item: (item.location.path, item.location.line, item.callee)
        )
        self._decorators.sort(
            key=lambda item: (item.location.path, item.location.line, item.expression)
        )
        self._issues.sort(key=lambda item: (item.path, item.kind, item.line or 0))

    def _safe_target(self, candidate: Path) -> bool:
        try:
            target = candidate.resolve(strict=True)
            relative_target = target.relative_to(self.root)
        except (OSError, ValueError):
            return False
        return not any(
            part in self.excluded_directories for part in relative_target.parts
        )

    def _index_file(self, candidate: Path) -> None:
        relative = candidate.relative_to(self.root).as_posix()
        if candidate.is_symlink() and not self._safe_target(candidate):
            self._issues.append(
                AnalysisIssue(relative, "unsafe_symlink", "symbolic link escapes repository")
            )
            return
        self._files.append(relative)
        try:
            size = candidate.stat().st_size
        except OSError as exception:
            self._issues.append(AnalysisIssue(relative, "read_error", str(exception)))
            return
        if size > self.max_file_size:
            self._oversized.add(relative)
            self._issues.append(
                AnalysisIssue(
                    relative,
                    "oversized",
                    f"source exceeds {self.max_file_size} bytes",
                )
            )
            return
        try:
            with tokenize.open(candidate) as source_file:
                source = source_file.read()
        except (OSError, UnicodeError, SyntaxError) as exception:
            self._issues.append(AnalysisIssue(relative, "read_error", str(exception)))
            return
        self._sources[relative] = source
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exception:
            self._issues.append(
                AnalysisIssue(
                    relative,
                    "syntax_error",
                    exception.msg,
                    exception.lineno,
                )
            )
            return
        indexer = _Indexer(relative)
        indexer.visit(tree)
        self._symbols.extend(indexer.symbols)
        self._imports.extend(indexer.imports)
        self._calls.extend(indexer.calls)
        self._decorators.extend(indexer.decorators)
