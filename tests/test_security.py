"""
Tests that no Python source files use forbidden imports or calls.

The competition sandbox security scanner disqualifies any submission using:
  Blocked modules (submission files): os, subprocess, socket, ctypes, builtins,
    sys, importlib, pickle, marshal, shelve, shutil, yaml, requests, urllib, http,
    multiprocessing, threading, signal, gc, code, codeop, pty
  Blocked builtins: eval(), exec(), compile(), __import__()

Training files are scanned with the original narrower blocklist since they
legitimately use some modules (e.g. shutil in data conversion scripts).

These tests catch violations before they reach the sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# File collections
# ---------------------------------------------------------------------------

# Submission files: these are shipped to the competition sandbox
SUBMISSION_FILES_AT_ROOT = [REPO_ROOT / "run.py"]
SUBMISSION_DIRS = [REPO_ROOT / "src"]

# Training files: run only inside our own containers, narrower restrictions
TRAINING_DIRS = [REPO_ROOT / "training"]

# ---------------------------------------------------------------------------
# Blocklists
# ---------------------------------------------------------------------------

# Full extended blocklist for submission files (from competition security docs)
_SUBMISSION_BLOCKED_MODULES = (
    "os|subprocess|socket|ctypes|builtins|sys|importlib|pickle|marshal|shelve"
    "|shutil|yaml|requests|urllib|http|multiprocessing|threading|signal|gc"
    "|code|codeop|pty"
)

# Narrower blocklist for training files (these run in our container, not sandbox)
_TRAINING_BLOCKED_MODULES = "os|subprocess|socket|ctypes|builtins"

_BLOCKED_BUILTINS = "eval|exec|compile|__import__"


def _make_patterns(blocked_modules: str) -> list[tuple[re.Pattern[str], str]]:
    """Build forbidden-pattern list for a given module blocklist."""
    return [
        (re.compile(rf"^\s*import ({blocked_modules})\b", re.MULTILINE), "blocked module import"),
        (
            re.compile(rf"^\s*from ({blocked_modules})(\s|\.|;|$)", re.MULTILINE),
            "blocked from-import",
        ),
        (re.compile(rf"^\s*({_BLOCKED_BUILTINS})\s*\(", re.MULTILINE), "blocked builtin call"),
    ]


SUBMISSION_PATTERNS = _make_patterns(_SUBMISSION_BLOCKED_MODULES)
TRAINING_PATTERNS = _make_patterns(_TRAINING_BLOCKED_MODULES)


def _collect_submission_files() -> list[Path]:
    files: list[Path] = list(SUBMISSION_FILES_AT_ROOT)
    for d in SUBMISSION_DIRS:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(set(files))


def _collect_training_files() -> list[Path]:
    files: list[Path] = []
    for d in TRAINING_DIRS:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(set(files))


def _find_violations(
    path: Path, patterns: list[tuple[re.Pattern[str], str]]
) -> list[tuple[str, int, str]]:
    """Return list of (pattern_name, line_number, matched_line) for any violation."""
    violations: list[tuple[str, int, str]] = []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    for pattern, name in patterns:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            matched_line = lines[line_no - 1].strip()
            violations.append((name, line_no, matched_line))
    return violations


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "py_file",
    _collect_submission_files(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_submission_no_forbidden_imports(py_file: Path) -> None:
    """Submission files must not use any module from the extended competition blocklist."""
    violations = _find_violations(py_file, SUBMISSION_PATTERNS)
    if violations:
        details = "\n".join(f"  Line {ln}: {name!r} → {line}" for name, ln, line in violations)
        pytest.fail(
            f"Forbidden import in {py_file.relative_to(REPO_ROOT)}:\n{details}\n"
            "Fix: use pathlib.Path instead of os.path; remove blocked modules entirely."
        )


@pytest.mark.parametrize(
    "py_file",
    _collect_training_files(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_training_no_forbidden_imports(py_file: Path) -> None:
    """Training files must not use core blocked modules."""
    violations = _find_violations(py_file, TRAINING_PATTERNS)
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
