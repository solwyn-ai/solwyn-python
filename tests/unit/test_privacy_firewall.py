"""Structural firewall tests for the SDK prompt-privacy promise.

These tests enforce that customer prompts are:
  1. never passed into a log statement
  2. never materialized outside the narrow tokenizer scope
  3. never named with a variable name that might accidentally be logged
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SDK_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "solwyn"


def _iter_source_files() -> list[Path]:
    return [p for p in SDK_SRC.rglob("*.py") if "_privacy" not in p.name]


@pytest.mark.unit
def test_no_logger_calls_receive_prompt_variables() -> None:
    """Source files must not pass variables named `text`, `content`,
    `messages`, `system`, `prompt`, or `contents` into a logger call."""
    patterns = [
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\btext\b"),
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\bcontent\b"),
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\bmessages\b"),
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\bsystem\b"),
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\bprompt\b"),
        re.compile(r"logger\.(debug|info|warning|error|exception)\s*\([^)]*\bcontents\b"),
    ]
    violations: list[str] = []
    for path in _iter_source_files():
        source = path.read_text()
        for line_no, line in enumerate(source.splitlines(), start=1):
            for pat in patterns:
                if pat.search(line):
                    violations.append(f"{path.relative_to(SDK_SRC)}:{line_no}: {line.strip()}")
    assert not violations, (
        "Privacy violation candidates found — logger calls must never "
        "receive prompt/content/message variables:\n" + "\n".join(violations)
    )


@pytest.mark.unit
def test_extract_text_from_kwargs_is_removed() -> None:
    """The original _extract_text_from_kwargs that materialized the full
    joined prompt string must no longer exist."""
    client_py = (SDK_SRC / "client.py").read_text()
    assert "_extract_text_from_kwargs" not in client_py, (
        "client.py must not define or call _extract_text_from_kwargs — "
        "use solwyn._privacy.estimate_content_length instead."
    )


@pytest.mark.unit
def test_privacy_module_has_warning_banner() -> None:
    """The privacy-sensitive module must open with a visible banner."""
    privacy_py = (SDK_SRC / "_privacy.py").read_text()
    assert "PRIVACY" in privacy_py[:500], (
        "_privacy.py must open with a PRIVACY banner in the first 500 "
        "chars so contributors know not to log anything defined there."
    )


@pytest.mark.unit
def test_privacy_module_has_no_logging_import() -> None:
    """_privacy.py must never import the logging module."""
    privacy_py = (SDK_SRC / "_privacy.py").read_text()
    assert "import logging" not in privacy_py, (
        "_privacy.py must not import logging — prompt-adjacent code "
        "must never have access to a logger."
    )


@pytest.mark.unit
def test_no_print_calls_in_sdk_source() -> None:
    """SDK source files must not contain print() calls."""
    violations: list[str] = []
    for path in SDK_SRC.rglob("*.py"):
        source = path.read_text()
        for line_no, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("print(") or " print(" in stripped:
                if stripped.startswith("#"):
                    continue
                violations.append(f"{path.relative_to(SDK_SRC)}:{line_no}: {stripped}")
    assert not violations, (
        "SDK source must not contain print() calls — use logging instead "
        "(but never for prompt content):\n" + "\n".join(violations)
    )
