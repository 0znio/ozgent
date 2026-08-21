"""End-to-end tests driving a real worker subprocess over a real pipe.

These deliberately do not import the worker in-process: the thing worth
testing is the protocol as the Rust host actually sees it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

DEMO_TOOLS = textwrap.dedent(
    '''
    import asyncio
    from typing import Annotated, Literal
    from ozgent_tools import tool, ToolError

    @tool
    async def slow_echo(text: Annotated[str, "what to echo"], delay: float = 0.0) -> dict:
        """Echo text back after an optional delay."""
        await asyncio.sleep(delay)
        return {"echoed": text}

    @tool(description="Add two numbers.")
    def add(a: int, b: int = 1, mode: Literal["sum", "diff"] = "sum") -> int:
        print("a stray print must not corrupt the protocol")
        return a + b if mode == "sum" else a - b

    @tool
    def always_fails() -> str:
        """Always raises a ToolError."""
        raise ToolError("nope, cannot do that", retryable=True)

    @tool
    def crashes() -> str:
        """Raises an unexpected exception."""
        raise RuntimeError("boom")
    '''
)


class WorkerHarness:
    """Spawns the worker and speaks JSON-RPC to it."""

    def __init__(self, tool_dir: Path):
        env = dict(os.environ, PYTHONPATH=str(PKG_ROOT), PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ozgent_tools.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(PKG_ROOT),
        )
        self._next_id = 0
        self.tool_dir = tool_dir

    def send(self, method: str, params: dict | None = None) -> int:
        self._next_id += 1
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            req["params"] = params
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req).encode() + b"\n")
        self.proc.stdin.flush()
        return self._next_id

    def recv(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError(f"worker closed the pipe. stderr:\n{self.stderr_text()}")
        return json.loads(line)

    def call(self, method: str, params: dict | None = None) -> dict:
        """Send one request and read its reply, asserting the ids line up."""
        req_id = self.send(method, params)
        msg = self.recv()
        assert msg["id"] == req_id, f"id mismatch: sent {req_id}, got {msg['id']}"
        return msg

    def initialize(self, **params) -> dict:
        params.setdefault("tool_paths", [str(self.tool_dir)])
        return self.call("initialize", params)["result"]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("shutdown")
            except (BrokenPipeError, ValueError):
                pass
        try:
            # Closing stdin gives the worker EOF as a second exit path, in case
            # the shutdown request never landed.
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                if stream and not stream.closed:
                    stream.close()

    def stderr_text(self) -> str:
        """Drain stderr. Only safe once the process has exited."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        return self.proc.stderr.read().decode() if self.proc.stderr else ""


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tool_dir = Path(cls._tmp.name)
        (cls.tool_dir / "demo.py").write_text(DEMO_TOOLS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.w = WorkerHarness(self.tool_dir)
        self.addCleanup(self.w.close)

    # ------------------------------------------------------------ discovery

    def test_initialize_reports_builtins_and_user_tools(self):
        result = self.w.initialize()
        names = {t["name"] for t in result["tools"]}
        self.assertIn("web_search", names, "built-in tools must load")
        self.assertIn("slow_echo", names, "user tool directory must be scanned")
        self.assertEqual(result["errors"], [], "clean tool set must load without errors")
        self.assertEqual(result["protocol_version"], 1)

    def test_schema_is_derived_from_the_signature(self):
        result = self.w.initialize()
        add = next(t for t in result["tools"] if t["name"] == "add")
        schema = add["input_schema"]

        self.assertEqual(schema["properties"]["a"], {"type": "integer"})
        self.assertEqual(schema["properties"]["b"], {"type": "integer", "default": 1})
        self.assertEqual(schema["properties"]["mode"]["enum"], ["sum", "diff"])
        self.assertEqual(schema["required"], ["a"], "only a has no default")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(add["description"], "Add two numbers.")

    def test_annotated_metadata_becomes_a_description(self):
        result = self.w.initialize()
        echo = next(t for t in result["tools"] if t["name"] == "slow_echo")
        self.assertEqual(
            echo["input_schema"]["properties"]["text"]["description"], "what to echo"
        )

    def test_docstring_is_used_when_no_description_given(self):
        result = self.w.initialize()
        echo = next(t for t in result["tools"] if t["name"] == "slow_echo")
        self.assertEqual(echo["description"], "Echo text back after an optional delay.")

    def test_disabled_tools_are_not_exposed(self):
        result = self.w.initialize(disabled=["add", "crashes"])
        names = {t["name"] for t in result["tools"]}
        self.assertNotIn("add", names)
        self.assertNotIn("crashes", names)
        self.assertIn("slow_echo", names)

    # ----------------------------------------------------------------- calls

    def test_successful_call(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 2, "b": 3}})
        self.assertEqual(msg["result"], 5)

    def test_defaults_are_applied(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 10}})
        self.assertEqual(msg["result"], 11, "b must default to 1")

    def test_integral_float_is_accepted_for_an_integer(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 2.0, "b": 3.0}})
        self.assertEqual(msg["result"], 5, "models emit 2.0 where 2 is meant")

    # ------------------------------------------------------------- failures

    def test_bad_enum_is_rejected_before_the_tool_runs(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 1, "mode": "nope"}})
        self.assertEqual(msg["error"]["code"], -32602)
        self.assertIn("'sum', 'diff'", msg["error"]["message"])

    def test_missing_required_argument_is_rejected(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"b": 1}})
        self.assertEqual(msg["error"]["code"], -32602)
        self.assertIn("missing required", msg["error"]["message"])

    def test_unknown_argument_is_rejected(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 1, "colour": "red"}})
        self.assertEqual(msg["error"]["code"], -32602)
        self.assertIn("colour", msg["error"]["message"])

    def test_tool_error_is_distinguishable_and_carries_retryable(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "always_fails", "arguments": {}})
        self.assertEqual(msg["error"]["code"], -32000)
        self.assertTrue(msg["error"]["data"]["retryable"])

    def test_unexpected_exception_becomes_an_internal_error_with_traceback(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "crashes", "arguments": {}})
        self.assertEqual(msg["error"]["code"], -32603)
        self.assertIn("RuntimeError", msg["error"]["message"])
        self.assertIn("boom", msg["error"]["data"]["traceback"])

    def test_unknown_tool_lists_what_is_available(self):
        self.w.initialize()
        msg = self.w.call("call", {"name": "nope", "arguments": {}})
        self.assertEqual(msg["error"]["code"], -32000)
        self.assertIn("web_search", msg["error"]["message"])

    def test_unknown_method(self):
        self.w.initialize()
        msg = self.w.call("frobnicate")
        self.assertEqual(msg["error"]["code"], -32601)

    def test_malformed_json_does_not_kill_the_worker(self):
        self.w.initialize()
        assert self.w.proc.stdin is not None
        self.w.proc.stdin.write(b"{ this is not json\n")
        self.w.proc.stdin.flush()
        msg = self.w.recv()
        self.assertEqual(msg["error"]["code"], -32700)

        # The worker must still be usable afterwards.
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 1, "b": 1}})
        self.assertEqual(msg["result"], 2)

    def test_a_crashing_tool_does_not_kill_the_worker(self):
        self.w.initialize()
        self.w.call("call", {"name": "crashes", "arguments": {}})
        msg = self.w.call("call", {"name": "add", "arguments": {"a": 7}})
        self.assertEqual(msg["result"], 8)

    # ---------------------------------------------------------- concurrency

    def test_calls_run_concurrently(self):
        """Three 0.4s sleeps must overlap, not serialise."""
        self.w.initialize()
        started = time.monotonic()
        ids = [
            self.w.send("call", {"name": "slow_echo", "arguments": {"text": str(i), "delay": 0.4}})
            for i in range(3)
        ]
        replies = {self.w.recv()["id"] for _ in ids}
        elapsed = time.monotonic() - started

        self.assertEqual(replies, set(ids), "every call must be answered")
        self.assertLess(elapsed, 1.0, f"calls serialised: took {elapsed:.2f}s, expected ~0.4s")

    def test_a_slow_call_does_not_block_a_fast_one(self):
        self.w.initialize()
        slow = self.w.send("call", {"name": "slow_echo", "arguments": {"text": "slow", "delay": 0.5}})
        fast = self.w.send("call", {"name": "add", "arguments": {"a": 1}})

        first = self.w.recv()
        self.assertEqual(first["id"], fast, "the fast call must return first")
        second = self.w.recv()
        self.assertEqual(second["id"], slow)

    def test_cancel_stops_an_in_flight_call(self):
        self.w.initialize()
        call_id = "c-1"
        req = self.w.send(
            "call",
            {"name": "slow_echo", "call_id": call_id, "arguments": {"text": "x", "delay": 30}},
        )
        cancel_reply = self.w.call("cancel", {"call_id": call_id})
        self.assertTrue(cancel_reply["result"]["cancelled"])

        msg = self.w.recv()
        self.assertEqual(msg["id"], req)
        self.assertEqual(msg["error"]["code"], -32001)

    # ------------------------------------------------------------- protocol

    def test_stray_prints_go_to_stderr_not_the_protocol_stream(self):
        """`add` prints on every call; the stream must stay parseable."""
        self.w.initialize()
        for i in range(5):
            msg = self.w.call("call", {"name": "add", "arguments": {"a": i}})
            self.assertEqual(msg["result"], i + 1)

    def test_config_reaches_the_tool(self):
        result = self.w.initialize(
            config={"web_search": {"provider": "brave", "max_results": 3}}
        )
        self.assertIn("web_search", {t["name"] for t in result["tools"]})
        # An unconfigured provider must fail with a clear, actionable message
        # rather than a network error.
        msg = self.w.call(
            "call",
            {"name": "web_search", "arguments": {"query": "test", "provider": "brave"}},
        )
        self.assertEqual(msg["error"]["code"], -32000)
        self.assertIn("BRAVE_API_KEY", msg["error"]["message"])

    def test_a_broken_tool_file_is_reported_but_others_still_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "good.py").write_text(
                "from ozgent_tools import tool\n\n@tool\ndef fine() -> int:\n    '''ok'''\n    return 1\n"
            )
            (path / "broken.py").write_text("this is not valid python !!!\n")

            w = WorkerHarness(path)
            self.addCleanup(w.close)
            result = w.initialize(tool_paths=[str(path)])

            names = {t["name"] for t in result["tools"]}
            self.assertIn("fine", names, "a broken sibling must not block a good tool")
            self.assertTrue(any("broken.py" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
