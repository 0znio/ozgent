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
import contextvars
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
    #: What this tool does to the world: ``"read"``, ``"write"``, or
    #: ``"execute"``. ozgent asks the user before anything that is not a read,
    #: so a tool that quietly writes files while claiming to read is the one
    #: way to defeat the permission prompt. Unset means ``"unknown"``, which
    #: is asked about — silence must not be read as "harmless".
    effect: str = "unknown"
    #: The file this tool was defined in. Reported to the user so that
    #: "where does this tool come from" has an answer without grepping.
    source: str = ""

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Run the tool, off the event loop if it is synchronous.

        A blocking tool must not stall sibling calls, so sync functions go to
        the default executor rather than running inline.
        """
        if self.is_async:
            return await self.fn(**arguments)
        loop = asyncio.get_running_loop()
        # The context has to be carried across by hand: `run_in_executor` runs
        # the callable in a bare thread, so context variables set by the caller
        # — the flag saying the user approved this call — would be invisible to
        # every synchronous tool, which is most of the ones that need it.
        context = contextvars.copy_context()
        return await loop.run_in_executor(None, lambda: context.run(self.fn, **arguments))

    def spec(self) -> dict[str, Any]:
        """The wire form sent to the Rust side."""
        out = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "source": self.source,
            "effect": self.effect,
        }
        if self.output_schema is not None:
            out["output_schema"] = self.output_schema
        return out


def _source_of(fn: Callable[..., Any]) -> str:
    """The file a tool function was defined in, or "" if it has none.

    ``inspect.getfile`` raises for anything without one — a function built at
    the REPL or by ``exec`` — and a tool that declines to say where it lives
    is not a reason to refuse to register it.
    """
    try:
        return inspect.getfile(fn)
    except (TypeError, OSError):
        return ""


#: The effects a tool may declare. Anything else is a typo, and a typo that
#: silently became "unknown" would look like a working declaration while
#: quietly making ozgent ask about a tool the author meant to wave through.
EFFECTS = ("read", "write", "execute", "unknown")


def tool(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    output_schema: dict[str, Any] | None = None,
    effect: str = "unknown",
) -> Any:
    """Register a function as a tool.

    Usable bare (``@tool``) or called (``@tool(description=...)``). The
    description defaults to the function's docstring, since that is where a
    Python author writes it anyway.

    ``effect`` says what the tool does to the world, and decides whether
    ozgent runs it without asking:

    ``"read"``
        Looks something up and returns it. Runs unasked by default.
    ``"write"``
        Creates or changes a file, a record, or a remote resource.
    ``"execute"``
        Runs a program.

    Left unset it is ``"unknown"``, which ozgent asks about. That is the right
    default for a tool whose author has not thought about it, and it means
    every tool written before this existed keeps working — it simply asks.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        doc = description or inspect.getdoc(fn) or ""

        if effect not in EFFECTS:
            raise ValueError(
                f"{tool_name}: effect={effect!r} is not one of {', '.join(EFFECTS)}"
            )

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
            source=_source_of(fn),
            effect=effect,
        )
        REGISTRY[tool_name] = entry
        # The function stays directly callable so tools can be unit-tested
        # without going through the worker.
        fn.__ozgent_tool__ = entry  # type: ignore[attr-defined]
        return fn

    return wrap(_fn) if _fn is not None else wrap


#: Config sections that are not tools, such as ``permissions``. They arrive
#: through the same transport and would otherwise be dropped for having no
#: tool of that name.
SHARED_CONFIG: dict[str, dict[str, Any]] = {}


def get_config(tool_name: str) -> dict[str, Any]:
    """Settings for one tool, from ``[tools.config.<name>]`` in config.toml.

    Falls back to the shared sections, so ``permissions`` resolves even though
    nothing registers a tool by that name.
    """
    entry = REGISTRY.get(tool_name)
    if entry is not None:
        return entry.config
    return SHARED_CONFIG.get(tool_name, {})


class ToolError(Exception):
    """Raised for a failure the model should see and can act on.

    Anything else that escapes a tool is a bug, and is reported as an internal
    error rather than being fed back into the conversation as advice.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable
