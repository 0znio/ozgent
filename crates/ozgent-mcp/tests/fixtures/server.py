"""A minimal MCP server over stdio, for testing the client against.

Small on purpose: it implements exactly the handshake and the two methods
ozgent uses, plus the awkward cases worth proving — a paginated tool list, a
tool that reports its own failure, and one that never answers.

Not a mock. It is a real server speaking the real protocol down a real pipe, so
the test covers the framing, the ordering and the timeout as well as the
parsing.
"""

import json
import sys
import time

TOOLS_PAGE_ONE = [
    {
        "name": "echo",
        "description": "Give back what it was sent.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "wipe",
        "description": "Claims to change something.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": False},
    },
]

TOOLS_PAGE_TWO = [
    {
        "name": "structured",
        "description": "Returns structured output.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "explode",
        "description": "Reports its own failure.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "sleep",
        "description": "Never answers in time.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # No name: the client must drop this rather than offer something
    # it cannot call.
    {"description": "nameless"},
]


def reply(id_, result):
    send({"jsonrpc": "2.0", "id": id_, "result": result})


def error(id_, code, message):
    send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def call(id_, params):
    name = params.get("name")
    arguments = params.get("arguments") or {}

    if name == "echo":
        reply(id_, {"content": [{"type": "text", "text": arguments.get("text", "")}]})
    elif name == "structured":
        reply(
            id_,
            {
                "content": [{"type": "text", "text": "two things"}],
                "structuredContent": {"count": 2, "items": ["a", "b"]},
            },
        )
    elif name == "explode":
        reply(id_, {"isError": True, "content": [{"type": "text", "text": "it went wrong"}]})
    elif name == "sleep":
        time.sleep(30)
        reply(id_, {"content": []})
    elif name == "wipe":
        reply(id_, {"content": [{"type": "text", "text": "wiped"}]})
    else:
        error(id_, -32602, f"no tool called {name}")


def main():
    # Servers really do print to stdout; the client must survive it rather
    # than treating the line as protocol and giving up.
    sys.stdout.write("starting up\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue

        method = message.get("method")
        id_ = message.get("id")

        # A notification. Answering one would be a protocol error.
        if id_ is None:
            continue

        if method == "initialize":
            reply(
                id_,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fixture", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            if cursor is None:
                reply(id_, {"tools": TOOLS_PAGE_ONE, "nextCursor": "page2"})
            else:
                reply(id_, {"tools": TOOLS_PAGE_TWO})
        elif method == "tools/call":
            call(id_, message.get("params") or {})
        else:
            error(id_, -32601, f"no method called {method}")


if __name__ == "__main__":
    main()
