"""The tool worker process.

Speaks newline-delimited JSON-RPC 2.0 over stdio. One request per line, one
response per line, correlated by ``id``. Requests are dispatched concurrently,
so a slow tool never blocks its siblings.

Run as ``python -m ozgent_tools.worker``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION, __version__
from .base import REGISTRY, Tool, ToolError
from .validate import ValidationError, coerce

# JSON-RPC error codes. The negative range below -32000 is reserved for us.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TOOL_ERROR = -32000
TOOL_CANCELLED = -32001


class Protocol:
    """Framing over a private copy of stdout.

    Two details matter here.

    First, tool authors print for debugging, and a stray ``print`` on the real
    stdout would corrupt the stream mid-conversation. So the worker duplicates
    the original stdout, keeps that for protocol traffic only, and rebinds
    ``sys.stdout`` to stderr where the host can log it harmlessly.

    Second, stdin is read on a dedicated daemon thread rather than the default
    executor. A thread parked in ``readline`` never returns, and
    ``asyncio.run`` joins the default executor on the way out, so using it here
    would hang the process on every clean shutdown.
    """

    def __init__(self) -> None:
        self._fd = os.dup(sys.stdout.fileno())
        self._out = os.fdopen(self._fd, "wb", buffering=0)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr

        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        """Begin pumping stdin. Must be called from inside the event loop.

        The stdout capture in ``__init__`` deliberately happens earlier, before
        any tool module is imported, so that a print at import time is caught
        too.
        """
        self._loop = asyncio.get_running_loop()
        self._reader = threading.Thread(target=self._pump, name="stdin", daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        """Feed stdin lines into the event loop until EOF."""
        try:
            assert self._loop is not None
            for line in iter(sys.stdin.buffer.readline, b""):
                text = line.decode("utf-8", errors="replace").strip()
                self._loop.call_soon_threadsafe(self._queue.put_nowait, text)
        except (ValueError, OSError):
            pass  # stdin closed underneath us
        finally:
            try:
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            except RuntimeError:
                pass  # loop already gone

    async def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False, default=str).encode() + b"\n"
        async with self._lock:
            self._out.write(line)

    async def read(self) -> str | None:
        """Next line, or None at EOF."""
        return await self._queue.get()

class Worker:
    def __init__(self) -> None:
        self.proto = Protocol()
        self.running: dict[str, asyncio.Task[Any]] = {}
        self.shutdown = asyncio.Event()
        self.initialized = False

    # ---------------------------------------------------------------- loading

    def load_tools(self, paths: list[Path], disabled: list[str]) -> list[str]:
        """Import built-ins, then every ``*.py`` under each search path.

        Returns the list of load errors; one bad file must not prevent the
        other tools from working.
        """
        errors: list[str] = []

        try:
            from . import builtin

            for mod in builtin.discover():
                try:
                    importlib.import_module(f".builtin.{mod}", package=__package__)
                except Exception as exc:
                    errors.append(f"builtin {mod}: {exc}")
        except Exception as exc:
            errors.append(f"builtins: {exc}")

        for base in paths:
            if not base.is_dir():
                continue
            for file in sorted(base.rglob("*.py")):
                if file.name.startswith("_"):
                    continue
                try:
                    self._load_file(file)
                except Exception as exc:
                    errors.append(f"{file}: {exc}")

        for name in disabled:
            REGISTRY.pop(name, None)

        return errors

    @staticmethod
    def _load_file(file: Path) -> None:
        # A unique module name keeps two same-named files in different search
        # paths from shadowing each other in sys.modules.
        mod_name = f"ozgent_user_tool_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise

    # --------------------------------------------------------------- dispatch

    async def handle(self, request: dict[str, Any]) -> None:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                result = self.on_initialize(params)
            elif method == "list_tools":
                result = {"tools": [t.spec() for t in REGISTRY.values()]}
            elif method == "call":
                result = await self.on_call(params, req_id)
            elif method == "cancel":
                result = self.on_cancel(params)
            elif method == "shutdown":
                self.shutdown.set()
                result = {"ok": True}
            elif method == "ping":
                result = {"ok": True}
            else:
                await self.error(req_id, METHOD_NOT_FOUND, f"unknown method {method!r}")
                return
        except ValidationError as exc:
            await self.error(req_id, INVALID_PARAMS, str(exc))
            return
        except ToolError as exc:
            await self.error(
                req_id, TOOL_ERROR, str(exc), {"retryable": exc.retryable}
            )
            return
        except asyncio.CancelledError:
            await self.error(req_id, TOOL_CANCELLED, "cancelled")
            return
        except Exception as exc:
            await self.error(
                req_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}",
                {"traceback": traceback.format_exc()},
            )
            return

        if req_id is not None:
            await self.proto.send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def on_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        paths = [Path(p) for p in params.get("tool_paths", [])]
        disabled = list(params.get("disabled", []))
        errors = self.load_tools(paths, disabled)

        # Per-tool settings arrive as opaque data from config.toml.
        for name, cfg in (params.get("config") or {}).items():
            if name in REGISTRY and isinstance(cfg, dict):
                REGISTRY[name].config = cfg

        self.initialized = True
        return {
            "protocol_version": PROTOCOL_VERSION,
            "worker_version": __version__,
            "python": sys.version.split()[0],
            "tools": [t.spec() for t in REGISTRY.values()],
            "errors": errors,
        }

    async def on_call(self, params: dict[str, Any], req_id: Any) -> Any:
        name = params.get("name")
        entry: Tool | None = REGISTRY.get(name)
        if entry is None:
            known = ", ".join(sorted(REGISTRY)) or "none"
            raise ToolError(f"no tool named {name!r}. Available: {known}")

        arguments = coerce(params.get("arguments") or {}, entry.input_schema)

        task = asyncio.ensure_future(entry.invoke(arguments))
        key = str(params.get("call_id") or req_id)
        self.running[key] = task
        try:
            return await task
        finally:
            self.running.pop(key, None)

    def on_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        key = str(params.get("call_id", ""))
        task = self.running.get(key)
        if task is not None:
            task.cancel()
        return {"cancelled": task is not None}

    async def error(
        self, req_id: Any, code: int, message: str, data: Any = None
    ) -> None:
        if req_id is None:
            return
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        await self.proto.send({"jsonrpc": "2.0", "id": req_id, "error": err})

    # ------------------------------------------------------------------- loop

    async def serve(self) -> None:
        self.proto.start()
        pending: set[asyncio.Task[Any]] = set()
        stop = asyncio.ensure_future(self.shutdown.wait())

        while True:
            read = asyncio.ensure_future(self.proto.read())
            done, _ = await asyncio.wait(
                {read, stop}, return_when=asyncio.FIRST_COMPLETED
            )
            if stop in done:
                read.cancel()
                break

            line = read.result()
            if line is None:
                break  # host closed the pipe
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                await self.error(0, PARSE_ERROR, f"invalid JSON: {exc}")
                continue
            if not isinstance(request, dict):
                await self.error(0, INVALID_REQUEST, "request must be a JSON object")
                continue

            task = asyncio.ensure_future(self.handle(request))
            pending.add(task)
            task.add_done_callback(pending.discard)

        stop.cancel()
        for task in list(pending):
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="ozgent-tool-worker")
    parser.add_argument("--tool-path", action="append", default=[], type=Path)
    args = parser.parse_args()

    worker = Worker()
    if args.tool_path:
        # Convenience for running the worker by hand; the host normally passes
        # search paths through `initialize` instead.
        worker.load_tools(args.tool_path, [])

    try:
        asyncio.run(worker.serve())
    except KeyboardInterrupt:
        return 130

    # Exit without running interpreter finalization.
    #
    # The stdin reader is a daemon thread parked in `readline`, holding the
    # buffered reader's lock. Finalization tries to acquire that lock and dies
    # with a fatal error. There is nothing left to clean up at this point, so
    # skipping finalization is both safe and quiet.
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
