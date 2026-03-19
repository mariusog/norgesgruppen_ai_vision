"""
Tests that no Python source files use forbidden imports or calls.

The competition sandbox security scanner disqualifies any submission using:
  Blocked modules: os, subprocess, socket, ctypes, builtins
  Blocked builtins: eval(), exec(), compile(), __import__()

These tests catch violations before they reach the sandbox.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# All Python source directories to scan
SOURCE_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "training",
]
SOURCE_FILES_AT_ROOT = [
    REPO_ROOT / "run.py",
]

_BLOCKED_MODULES = "os|subprocess|socket|ctypes|builtins"
_BLOCKED_BUILTINS = "eval|exec|compile|__import__"

FORBIDDEN_PATTERNS = [
    # Blocked module imports (anchored to line start)
    (re.compile(rf"^\s*import ({_BLOCKED_MODULES})\b", re.MULTILINE), "blocked module import"),
    (re.compile(rf"^\s*from ({_BLOCKED_MODULES})(\s|\.|;|$)", re.MULTILINE), "blocked from-import"),
    # Blocked builtin calls (anchored to line start to avoid matching comments/strings)
    (re.compile(rf"^\s*({_BLOCKED_BUILTINS})\s*\(", re.MULTILINE), "blocked builtin call"),
]


def _collect_python_files() -> list[Path]:
    files: list[Path] = list(SOURCE_FILES_AT_ROOT)
    for source_dir in SOURCE_DIRS:
        if source_dir.exists():
            files.extend(source_dir.rglob("*.py"))
    return sorted(set(files))


def _find_violations(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (pattern_name, line_number, matched_line) for any violation."""
    violations: list[tuple[str, int, str]] = []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    for pattern, name in FORBIDDEN_PATTERNS:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            matched_line = lines[line_no - 1].strip()
            violations.append((name, line_no, matched_line))
    return violations


def _python_files() -> list[Path]:
    return _collect_python_files()


@pytest.mark.parametrize("py_file", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_forbidden_imports(py_file: Path) -> None:
    """Every Python file must not use os, subprocess, or socket imports."""
    violations = _find_violations(py_file)
    if violations:
        details = "\n".join(f"  Line {ln}: {name!r} → {line}" for name, ln, line in violations)
        pytest.fail(
            f"Forbidden import in {py_file.relative_to(REPO_ROOT)}:\n{details}\n"
            "Fix: use pathlib.Path instead of os.path; avoid subprocess and socket entirely."
        )


def test_run_py_exists_at_root() -> None:
    """run.py must exist at the repository root (competition requirement)."""
    assert (REPO_ROOT / "run.py").exists(), "run.py must exist at the repo root"


def test_run_py_has_required_arguments() -> None:
    """run.py must accept --input and --output arguments."""
    run_content = (REPO_ROOT / "run.py").read_text()
    assert "--input" in run_content, "run.py must define --input argument"
    assert "--output" in run_content, "run.py must define --output argument"


def test_run_py_uses_torch_no_grad() -> None:
    """run.py must use torch.no_grad() to stay within memory limits."""
    run_content = (REPO_ROOT / "run.py").read_text()
    assert "torch.no_grad()" in run_content, "run.py must wrap inference in torch.no_grad()"


def test_run_py_uses_pathlib() -> None:
    """run.py must use pathlib for file operations."""
    run_content = (REPO_ROOT / "run.py").read_text()
    assert "from pathlib import Path" in run_content or "import pathlib" in run_content, (
        "run.py must use pathlib for file operations (not os.path)"
    )
