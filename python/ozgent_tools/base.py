"""The tool authoring API.

Defining a tool is defining a function::

    from ozgent_tools import tool

    @tool(description="Look something up on the web.")
    async def web_search(query: str, count: int = 5) -> dict:
        ...

The decorator reads the signature to build the model-facing schema, so there
is no separate manifest to keep in sync.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import schema_from_signature

# Populated at import time by the decorator; the worker reads it after loading
# every tool module.
REGISTRY: dict[str, "Tool"] = {}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    is_async: bool = False
    # Settings from ``[tools.config.<name>]``, injected before the first call.
    config: dict[str, Any] = field(default_factory=dict)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Run the tool, off the event loop if it is synchronous.

        A blocking tool must not stall sibling calls, so sync functions go to
        the default executor rather than running inline.
        """
        if self.is_async:
            return await self.fn(**arguments)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.fn(**arguments))

    def spec(self) -> dict[str, Any]:
        """The wire form sent to the Rust side."""
        out = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if self.output_schema is not None:
            out["output_schema"] = self.output_schema
        return out


def tool(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Any:
    """Register a function as a tool.

    Usable bare (``@tool``) or called (``@tool(description=...)``). The
    description defaults to the function's docstring, since that is where a
    Python author writes it anyway.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        doc = description or inspect.getdoc(fn) or ""

        if tool_name in REGISTRY:
            raise ValueError(
                f"duplicate tool name {tool_name!r}: already defined in "
                f"{inspect.getmodule(REGISTRY[tool_name].fn)}"
            )

        entry = Tool(
            name=tool_name,
            description=doc.strip(),
            fn=fn,
            input_schema=schema_from_signature(fn),
            output_schema=output_schema,
            is_async=inspect.iscoroutinefunction(fn),
        )
        REGISTRY[tool_name] = entry
        # The function stays directly callable so tools can be unit-tested
        # without going through the worker.
        fn.__ozgent_tool__ = entry  # type: ignore[attr-defined]
        return fn

    return wrap(_fn) if _fn is not None else wrap


def get_config(tool_name: str) -> dict[str, Any]:
    """Settings for one tool, from ``[tools.config.<name>]`` in config.toml."""
    entry = REGISTRY.get(tool_name)
    return entry.config if entry else {}


class ToolError(Exception):
    """Raised for a failure the model should see and can act on.

    Anything else that escapes a tool is a bug, and is reported as an internal
    error rather than being fed back into the conversation as advice.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable
