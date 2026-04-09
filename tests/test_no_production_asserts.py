"""Production SDK code must not use ``assert`` for runtime validation."""

from __future__ import annotations

import ast
from pathlib import Path

SDK_SRC = Path(__file__).parent.parent / "src" / "solwyn"


def test_no_assert_statements_in_production_code() -> None:
    violations: list[str] = []
    for path in SDK_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                violations.append(f"{path.relative_to(SDK_SRC)}:{node.lineno}")
    assert not violations, (
        "The SDK must not use `assert` for runtime validation "
        "(stripped under python -O). Replace with `raise RuntimeError`:\n" + "\n".join(violations)
    )
