"""Derive JSON Schema from ordinary Python type hints.

Writing a tool should mean writing a function. Everything the model needs to
know about its arguments is already in the signature, so we read it from there
rather than making the author repeat it in a schema literal.
"""

from __future__ import annotations

import enum
import inspect
import types
import typing
from typing import Any, Annotated, Literal, get_args, get_origin

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# Sentinel distinguishing "no default" from "default is None".
_MISSING = object()


class SchemaError(TypeError):
    """A tool signature that cannot be expressed as JSON Schema."""


def _unwrap_annotated(hint: Any) -> tuple[Any, str | None]:
    """Split ``Annotated[T, "docs"]`` into ``T`` and the first string metadata."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        base = args[0]
        description = next((m for m in args[1:] if isinstance(m, str)), None)
        return base, description
    return hint, None


def _is_optional(hint: Any) -> tuple[Any, bool]:
    """Reduce ``T | None`` to ``T`` plus a nullable flag."""
    origin = get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(hint) if a is not type(None)]
        nullable = len(args) != len(get_args(hint))
        if not args:
            raise SchemaError("a parameter typed only None cannot be represented")
        if len(args) == 1:
            return args[0], nullable
        # A genuine multi-type union: keep it, handled by type_to_schema.
        return typing.Union[tuple(args)], nullable  # type: ignore[return-value]
    return hint, False


def type_to_schema(hint: Any) -> dict[str, Any]:
    """Convert a single type hint into a JSON Schema fragment."""
    hint, description = _unwrap_annotated(hint)
    hint, nullable = _is_optional(hint)
    # The inner type may itself be Annotated, e.g. Optional[Annotated[int, "n"]].
    hint, inner_desc = _unwrap_annotated(hint)
    description = description or inner_desc

    schema = _core_schema(hint)
    if nullable and "type" in schema and isinstance(schema["type"], str):
        schema["type"] = [schema["type"], "null"]
    if description:
        schema["description"] = description
    return schema


def _core_schema(hint: Any) -> dict[str, Any]:
    if hint is Any or hint is inspect.Parameter.empty:
        return {}
    if hint in _PRIMITIVES:
        return {"type": _PRIMITIVES[hint]}
    if hint is type(None):
        return {"type": "null"}

    origin = get_origin(hint)

    if origin is Literal:
        values = list(get_args(hint))
        types_seen = {type(v) for v in values}
        schema: dict[str, Any] = {"enum": values}
        if len(types_seen) == 1:
            only = next(iter(types_seen))
            if only in _PRIMITIVES:
                schema["type"] = _PRIMITIVES[only]
        return schema

    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return {"enum": [m.value for m in hint]}

    if origin in (list, set, frozenset, tuple):
        args = get_args(hint)
        # Homogeneous sequences only; a fixed-length tuple maps poorly to the
        # argument shapes models actually emit.
        item = args[0] if args and args[0] is not Ellipsis else Any
        return {"type": "array", "items": type_to_schema(item)}

    if origin is dict:
        args = get_args(hint)
        if args and args[0] is not str:
            raise SchemaError("dict keys must be str to be valid JSON")
        value = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": type_to_schema(value)}

    if origin is typing.Union or origin is types.UnionType:
        return {"anyOf": [type_to_schema(a) for a in get_args(hint)]}

    if hint in (dict, list):
        return {"type": "object" if hint is dict else "array"}

    raise SchemaError(f"cannot express {hint!r} as JSON Schema")


def schema_from_signature(fn: Any) -> dict[str, Any]:
    """Build the ``input_schema`` object for a tool function.

    Parameters with defaults become optional; everything else is required.
    ``*args`` and ``**kwargs`` are rejected, because the model has no way to
    know what to put in them.
    """
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception as exc:  # unresolvable forward reference
        raise SchemaError(f"cannot resolve type hints for {fn.__name__}: {exc}") from exc

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise SchemaError(
                f"{fn.__name__} uses *{name}; tools need an explicit parameter list"
            )

        hint = hints.get(name, Any)
        prop = type_to_schema(hint)

        default = param.default if param.default is not param.empty else _MISSING
        if default is _MISSING:
            required.append(name)
        elif _is_jsonable(default):
            prop["default"] = default

        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        # Models invent parameters; refusing them early gives a clearer error
        # than letting the call fail inside the tool body.
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _is_jsonable(v: Any) -> bool:
    if isinstance(v, (str, int, float, bool, type(None))):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_jsonable(i) for i in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_jsonable(i) for k, i in v.items())
    return False
