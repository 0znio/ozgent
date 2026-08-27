"""The boundaries around the acting tools.

These are the tests that matter most in this package. Everything else fails
visibly; a permission check that is subtly wrong fails by letting the model
do something the user never agreed to.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools import base
from ozgent_tools.base import ToolError
from ozgent_tools.builtin.fetch_url import fetch_url
from ozgent_tools.builtin.list_dir import list_dir
from ozgent_tools.builtin.run_command import run_command
from ozgent_tools.permissions import SECTION, check_command, check_host, resolve_within


def call(tool, **kwargs):
    """Invoke a registered tool the way the worker does."""
    fn = base.REGISTRY[tool].fn if tool in base.REGISTRY else tool
    return asyncio.run(fn(**kwargs))


class Permissions(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve()
        (self.root / "src").mkdir()
        (self.root / "src" / "main.rs").write_text("fn main() {}\n")
        (self.root / "README.md").write_text("hello\n")
        base.SHARED_CONFIG[SECTION] = {"root": str(self.root)}

    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)
        self.dir.cleanup()

    # ------------------------------------------------------------ filesystem

    def test_paths_outside_the_root_are_refused(self):
        with self.assertRaises(ToolError) as e:
            resolve_within("/etc/passwd", "list_dir")
        self.assertIn("outside the permitted directory", str(e.exception))

    def test_dot_dot_cannot_escape(self):
        # Resolution happens before the check precisely so this fails.
        with self.assertRaises(ToolError):
            resolve_within("../../../etc", "list_dir")

    def test_a_symlink_out_of_the_root_is_refused(self):
        link = self.root / "escape"
        os.symlink("/etc", link)
        with self.assertRaises(ToolError):
            resolve_within("escape", "list_dir")

    def test_listing_stays_inside_and_finds_files(self):
        out = call("list_dir", path=".", depth=2)
        names = " ".join(out["entries"])
        self.assertIn("README.md", names)
        self.assertIn("src/", names)
        self.assertIn("main.rs", names)

    # ----------------------------------------------------------------- shell

    def test_commands_are_refused_until_permitted(self):
        with self.assertRaises(ToolError) as e:
            check_command("ls")
        self.assertIn(f"[tools.config.{SECTION}] shell = true", str(e.exception))

    def test_permitted_but_empty_allowlist_permits_nothing(self):
        # "on" and "anything" must not be the same setting.
        base.SHARED_CONFIG[SECTION].update(shell=True)
        with self.assertRaises(ToolError) as e:
            check_command("ls")
        self.assertIn("no commands are allowed", str(e.exception))

    def test_only_allowlisted_programs_run(self):
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        self.assertEqual(check_command("ls -la"), ["ls", "-la"])
        with self.assertRaises(ToolError) as e:
            check_command("rm -rf /")
        self.assertIn("not in the allowlist", str(e.exception))

    def test_shell_metacharacters_are_refused(self):
        # An allowed program must not be able to introduce a disallowed one.
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        for attempt in ["ls; rm -rf /", "ls && rm x", "ls | sh", "ls > /etc/passwd", "ls `whoami`"]:
            with self.assertRaises(ToolError, msg=attempt):
                check_command(attempt)

    def test_an_allowlisted_program_reached_by_path_is_still_matched(self):
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        self.assertEqual(check_command("/bin/ls")[0], "/bin/ls")

    def test_a_path_to_a_forbidden_program_is_refused(self):
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        with self.assertRaises(ToolError):
            check_command("/bin/rm x")

    def test_running_a_permitted_command_returns_its_output(self):
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        out = call("run_command", command="ls")
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("README.md", out["stdout"])

    def test_a_failing_command_reports_its_code_rather_than_raising(self):
        # A non-zero exit is information for the model, not a tool failure.
        base.SHARED_CONFIG[SECTION].update(shell=True, shell_allow=["ls"])
        out = call("run_command", command="ls /nonexistent-ozgent-path")
        self.assertNotEqual(out["exit_code"], 0)
        self.assertTrue(out["stderr"].strip())

    # --------------------------------------------------------------- network

    def test_fetching_is_refused_until_permitted(self):
        with self.assertRaises(ToolError) as e:
            check_host("https://example.com/page")
        self.assertIn(f"[tools.config.{SECTION}] network = true", str(e.exception))

    def test_an_allowlist_restricts_hosts(self):
        base.SHARED_CONFIG[SECTION].update(network=True, network_allow=["example.com"])
        self.assertEqual(check_host("https://example.com/x"), "example.com")
        # Subdomains of an allowed host are allowed.
        self.assertEqual(check_host("https://docs.example.com/x"), "docs.example.com")
        with self.assertRaises(ToolError):
            check_host("https://evil.com/x")

    def test_a_lookalike_host_is_not_matched(self):
        base.SHARED_CONFIG[SECTION].update(network=True, network_allow=["example.com"])
        with self.assertRaises(ToolError):
            check_host("https://notexample.com/x")

    def test_non_http_schemes_are_refused(self):
        base.SHARED_CONFIG[SECTION].update(network=True)
        for url in ["file:///etc/passwd", "ftp://example.com/x"]:
            with self.assertRaises(ToolError, msg=url):
                check_host(url)


class PageToText(unittest.TestCase):
    """The floor: markup out, words in. See test_extract.py for the rest."""

    def test_script_and_markup_are_stripped(self):
        from ozgent_tools.extract import extract

        page = "<html><head><style>a{}</style><script>x()</script></head>" \
               "<body><h1>Title</h1><p>Body &amp; more</p></body></html>"
        text = extract(page)["text"]
        self.assertIn("Title", text)
        self.assertIn("Body & more", text)
        self.assertNotIn("x()", text)
        self.assertNotIn("<p>", text)


if __name__ == "__main__":
    unittest.main()
