"""Project ID and project API key validation.

Project key and project ID format validation.

Security features applied to every validator:
- Unicode NFC normalization to prevent homograph attacks
- ASCII-only enforcement to prevent encoding exploits
- Regex pattern validation for allowed characters
- Path traversal prevention (reject ``..``, ``/``, ``\\``)
"""

import re
import unicodedata
from typing import Final

PROJECT_ID_PATTERN: Final = re.compile(r"^proj_[a-f0-9]{24}$")
PROJECT_KEY_PATTERN: Final = re.compile(r"^sk_proj_[a-f0-9]{64}$")


def _security_checks(value: str, label: str) -> str:
    """Common security checks shared by all validators."""
    if not value:
        raise ValueError(f"{label} cannot be empty")

    value = unicodedata.normalize("NFC", value)

    if not value.isascii():
        raise ValueError(f"Invalid {label}: must contain only ASCII characters")

    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: path traversal patterns not allowed")

    return value


def validate_project_id(project_id: str) -> str:
    """Validate and return a project ID (canonical implementation)."""
    project_id = _security_checks(project_id, "project ID")

    if not PROJECT_ID_PATTERN.match(project_id):
        display = f"{project_id[:24]}..." if len(project_id) > 24 else project_id
        raise ValueError(
            f"Invalid project ID: must match proj_<24 lowercase hex chars>. Got: {display}"
        )

    return project_id


def validate_project_key_format(api_key: str) -> str:
    """Validate and return a project API key (format only, not authentication)."""
    api_key = _security_checks(api_key, "API key")

    if not PROJECT_KEY_PATTERN.match(api_key):
        display = f"{api_key[:12]}..." if len(api_key) > 12 else "<too short>"
        raise ValueError(
            f"Invalid API key format: must start with 'sk_proj_' and contain 64 hex chars. "
            f"Got: {display}"
        )

    return api_key
