"""A minimal MCP server over stdio, for testing the client against.

Small on purpose: it implements exactly the handshake and the two methods
ozgent uses, plus the awkward cases worth proving — a paginated tool list, a
tool that reports its own failure, and one that never answers.

Not a mock. It is a real server speaking the real protocol down a real pipe, so
the test covers the framing, the ordering and the timeout as well as the
parsing.
"""

import json
import os
import socket
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


# Listed only when asked for, so the tests that count tools are unaffected:
# what a hostile server would try, for the sandbox tests to watch fail.
PROBE = {
    "name": "probe",
    "description": "Try to read, write, connect and look at the environment.",
    "inputSchema": {"type": "object", "properties": {}},
}


def attempt(action):
    try:
        return f"ok: {action()}"
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        return f"refused: {type(exc).__name__}"


def probe(arguments):
    out = {}
    for path in arguments.get("read", []):
        out[f"read {path}"] = attempt(lambda p=path: open(p).read()[:40])
    for path in arguments.get("write", []):
        out[f"write {path}"] = attempt(lambda p=path: open(p, "w").write("x"))
    for host, port in arguments.get("connect", []):
        out[f"connect {host}:{port}"] = attempt(lambda h=host, p=port: socket.create_connection((h, p), timeout=3) and "connected")
    for host, port in arguments.get("udp", []):
        out[f"udp {host}:{port}"] = attempt(
            lambda h=host, p=port: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", (h, p)))
    for path in arguments.get("unix", []):
        out[f"unix {path}"] = attempt(lambda p=path: socket.socket(socket.AF_UNIX).connect(p) or "connected")
    for path in arguments.get("list", []):
        out[f"list {path}"] = attempt(lambda p=path: len(os.listdir(p)))
    for name in arguments.get("env", []):
        out[f"env {name}"] = os.environ.get(name, "<unset>")
    out["home"] = os.environ.get("HOME", "")
    return out


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
    elif name == "probe":
        reply(id_, {"content": [{"type": "text", "text": "probed"}], "structuredContent": probe(arguments)})
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
                extra = [PROBE] if os.environ.get("FIXTURE_PROBE") == "1" else []
                reply(id_, {"tools": TOOLS_PAGE_TWO + extra})
        elif method == "tools/call":
            call(id_, message.get("params") or {})
        else:
            error(id_, -32601, f"no method called {method}")


if __name__ == "__main__":
    main()
