"""Tests for public credential format validators."""

from __future__ import annotations

import pytest

import solwyn._validation as validation
from solwyn._validation import validate_project_id, validate_project_key_format


@pytest.mark.unit
def test_validate_project_key_format_accepts_project_key() -> None:
    key = "sk_proj_" + "a" * 64

    assert validate_project_key_format(key) == key


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "sk_solwyn_" + "a" * 64,
        "sk_proj_" + "A" * 64,
        "sk_proj_" + "a" * 63,
        "sk_proj_" + "a" * 65,
    ],
)
def test_validate_project_key_format_rejects_non_project_key_formats(key: str) -> None:
    with pytest.raises(ValueError, match="sk_proj_"):
        validate_project_key_format(key)


@pytest.mark.unit
def test_legacy_validate_api_key_format_symbol_is_removed() -> None:
    assert not hasattr(validation, "validate_api_key_format")


@pytest.mark.unit
def test_validate_project_id_accepts_public_copy_id() -> None:
    project_id = "proj_" + "a" * 24

    assert validate_project_id(project_id) == project_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "project_id",
    [
        "proj_" + "a" * 8,
        "proj_" + "A" * 24,
        "proj_" + "g" * 24,
        "proj_" + "a" * 25,
    ],
)
def test_validate_project_id_rejects_old_or_non_hex_formats(project_id: str) -> None:
    with pytest.raises(ValueError, match="24 lowercase hex"):
        validate_project_id(project_id)
