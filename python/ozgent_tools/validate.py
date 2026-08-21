"""A small JSON Schema checker for tool arguments.

Only the subset :mod:`ozgent_tools.schema` can emit is supported, which keeps
the runtime dependency-free. The point is to reject a malformed model-generated
call with a message the model can act on, before it reaches user code.
"""

from __future__ import annotations

from typing import Any

_CHECKS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


class ValidationError(ValueError):
    pass


def validate(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    """Raise :class:`ValidationError` if ``value`` does not match ``schema``."""
    if not schema:
        return

    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                validate(value, branch, path)
                return
            except ValidationError:
                continue
        raise ValidationError(f"{path} matches none of the permitted types")

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        raise ValidationError(f"{path} must be one of: {allowed} (got {value!r})")

    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        # An integer schema accepts 2.0 but not 2.5; models routinely emit the
        # former and rejecting it would be needlessly strict.
        if "integer" in options and isinstance(value, float) and value.is_integer():
            value = int(value)
        if not any(_CHECKS.get(o, lambda _v: True)(value) for o in options):
            raise ValidationError(
                f"{path} must be {' or '.join(options)}, got {_name(value)}"
            )

    if isinstance(value, dict) and schema.get("type") == "object":
        _validate_object(value, schema, path)
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    props: dict[str, Any] = schema.get("properties", {})

    missing = [k for k in schema.get("required", []) if k not in value]
    if missing:
        raise ValidationError(
            f"{path} is missing required {_plural(missing)}: {', '.join(missing)}"
        )

    if schema.get("additionalProperties") is False:
        unknown = [k for k in value if k not in props]
        if unknown:
            known = ", ".join(props) or "none"
            raise ValidationError(
                f"{path} has unknown {_plural(unknown)}: {', '.join(unknown)}. "
                f"Accepted: {known}"
            )

    for key, sub in props.items():
        if key in value:
            validate(value[key], sub, f"{path}.{key}")


def coerce(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise an arguments object.

    Fills in declared defaults and narrows integral floats, so the tool body
    receives exactly the types its annotations promise.
    """
    validate(value, schema)
    out = dict(value)
    for key, sub in schema.get("properties", {}).items():
        if key not in out:
            if "default" in sub:
                out[key] = sub["default"]
            continue
        types = sub.get("type")
        types = types if isinstance(types, list) else [types]
        if "integer" in types and isinstance(out[key], float) and out[key].is_integer():
            out[key] = int(out[key])
    return out


def _name(v: Any) -> str:
    return {bool: "boolean", type(None): "null", str: "string", int: "integer",
            float: "number", list: "array", dict: "object"}.get(type(v), type(v).__name__)


def _plural(items: list[str]) -> str:
    return "properties" if len(items) > 1 else "property"
