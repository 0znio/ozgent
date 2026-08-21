"""ozgent's Python tool runtime.

Tools run in a separate long-lived process and speak newline-delimited JSON-RPC
to the Rust host over stdio. Keeping them out-of-process means a tool that
hangs, leaks, or crashes cannot disturb inference.
"""

from .base import REGISTRY, Tool, ToolError, get_config, tool

__all__ = ["REGISTRY", "Tool", "ToolError", "get_config", "tool"]

PROTOCOL_VERSION = 1
__version__ = "0.1.0"
