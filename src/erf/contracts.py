"""Schema loading and validation for ERF catalog records."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker

_SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"
_validators: dict[str, Draft202012Validator] = {}


def _load_validator(name: str) -> Draft202012Validator:
    """Load and cache a compiled validator for a schema name (without .schema.json)."""
    if name not in _validators:
        path = _SCHEMAS_DIR / f"{name}.schema.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        schema = json.loads(path.read_text())
        _validators[name] = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        )
    return _validators[name]


def validate(record: dict[str, Any], schema_name: str) -> None:
    """Validate record against the named schema. Raises jsonschema.ValidationError on failure."""
    v = _load_validator(schema_name)
    v.validate(record)
    # Custom cross-field checks not expressible in JSON Schema alone
    _cross_field_checks(record, schema_name)


def _cross_field_checks(record: dict[str, Any], schema_name: str) -> None:
    if schema_name == "embedding":
        dim = record.get("dim")
        vec = record.get("vector")
        if dim is not None and vec is not None:
            if len(vec) != dim:
                raise jsonschema.ValidationError(
                    f"embedding vector length {len(vec)} != declared dim {dim}"
                )


def validate_all(records: list[dict[str, Any]], schema_name: str) -> list[str]:
    """Validate a list of records. Returns list of error messages (empty = all valid)."""
    errors = []
    for i, rec in enumerate(records):
        try:
            validate(rec, schema_name)
        except jsonschema.ValidationError as e:
            errors.append(f"record[{i}]: {e.message}")
    return errors


def schema_names() -> list[str]:
    """Return all available schema names."""
    return sorted(p.stem.replace(".schema", "") for p in _SCHEMAS_DIR.glob("*.schema.json"))
