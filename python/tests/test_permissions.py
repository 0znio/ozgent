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
from ozgent_tools.builtin.read_file import read_file
from ozgent_tools.builtin.run_command import run_command
from ozgent_tools.builtin.web_search import web_search
from ozgent_tools.builtin.write_file import write_file
from ozgent_tools.permissions import (
    SECTION,
    approving,
    check_command,
    check_host,
    resolve_within,
)


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


class ApprovalAtTheMomentOfTheCall(unittest.TestCase):
    """A yes the user gave in the moment counts as permission.

    The standing config answers "what may run with nobody looking". These
    tests pin the other half: a person who read the tool and its arguments and
    said yes has authorised that call, and a permission system that then
    refuses because a flag they never saw is off is arguing with its own user.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve()
        # Deliberately the most locked-down configuration there is: nothing
        # switched on, no allowlists. Approval alone has to carry every case.
        base.SHARED_CONFIG[SECTION] = {"root": str(self.root)}

    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)
        self.dir.cleanup()

    def test_a_flag_that_is_off_still_refuses_without_approval(self):
        with self.assertRaises(ToolError):
            check_command("git status")

    def test_approval_satisfies_the_flag_and_the_allowlist(self):
        with approving(True):
            self.assertEqual(check_command("git status"), ["git", "status"])

    def test_approval_reaches_outside_the_root(self):
        outside = Path(tempfile.gettempdir()).resolve() / "somewhere-else.txt"
        with self.assertRaises(ToolError):
            resolve_within(str(outside), "write_file")
        with approving(True):
            self.assertEqual(resolve_within(str(outside), "write_file"), outside)

    def test_approval_satisfies_the_network_allowlist(self):
        base.SHARED_CONFIG[SECTION]["network_allow"] = ["example.com"]
        with self.assertRaises(ToolError):
            check_host("https://other.test/page")
        with approving(True):
            self.assertEqual(check_host("https://other.test/page"), "other.test")

    def test_approval_does_not_lift_the_metacharacter_refusal(self):
        # Not a permission: commands run without a shell, so a pipe would not
        # do what the user reading it thinks it does. Approving a command must
        # not turn it into something else.
        with approving(True):
            with self.assertRaises(ToolError):
                check_command("cat /etc/passwd | mail me@example.com")

    def test_approval_does_not_leak_out_of_its_scope(self):
        with approving(True):
            pass
        with self.assertRaises(ToolError):
            check_command("git status")

    def test_approval_does_not_leak_between_concurrent_calls(self):
        """Calls share an event loop; one approval must not cover another.

        This is the hole a plain module-level flag would have: two tools in
        flight at once, one approved, and the other silently riding on it.
        """

        async def both():
            async def approved():
                with approving(True):
                    await asyncio.sleep(0)
                    return check_command("git status")

            async def unapproved():
                await asyncio.sleep(0)
                try:
                    check_command("rm -rf /")
                except ToolError:
                    return "refused"
                return "ran"

            return await asyncio.gather(approved(), unapproved())

        allowed, refused = asyncio.run(both())
        self.assertEqual(allowed, ["git", "status"])
        self.assertEqual(refused, "refused")


class ApprovalReachesEveryActingTool(unittest.TestCase):
    """The bug class: a tool that checks the permission flag itself.

    `write_file` read `perms()["write"]` directly instead of calling
    `require`, so an approval given at the prompt never reached the only code
    that was asking — pressing "yes" wrote nothing and said writing was not
    permitted. Nothing about that is specific to `write_file`; the next tool
    to inline the check would break the same way, silently.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve()
        # Nothing switched on. Approval alone has to carry every case.
        base.SHARED_CONFIG[SECTION] = {"root": str(self.root)}

    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)
        self.dir.cleanup()

    def test_the_decision_lives_in_one_place(self):
        """No builtin may read the permission flags for itself.

        A static check, because the behavioural ones below can only cover the
        tools that exist today. `perms()` is the accessor; reading it outside
        `permissions.py` means deciding outside `permissions.py`.
        """
        builtin = Path(base.__file__).parent / "builtin"
        offenders = [
            f.name
            for f in sorted(builtin.glob("*.py"))
            if "perms()" in f.read_text()
        ]
        self.assertEqual(
            offenders, [],
            "these decide permission for themselves and so cannot see an "
            "approval: use require()/resolve_within()/check_command()",
        )

    def test_writing_is_refused_unapproved_and_allowed_approved(self):
        target = str(self.root / "note.txt")
        with self.assertRaises(ToolError) as refused:
            call(write_file, path=target, content="hi")
        self.assertIn("permitted", str(refused.exception))

        async def approved():
            with approving(True):
                return await write_file(path=target, content="hi")

        result = asyncio.run(approved())
        self.assertEqual(result["bytes"], 2)
        self.assertTrue(Path(target).exists(), "approving must actually write the file")

    def test_running_a_command_is_refused_unapproved_and_allowed_approved(self):
        with self.assertRaises(ToolError):
            call(run_command, command="echo hi")

        async def approved():
            with approving(True):
                return await run_command(command="echo hi")

        self.assertEqual(asyncio.run(approved())["stdout"].strip(), "hi")


class DeclaredEffects(unittest.TestCase):
    """Every builtin says what it does, because the prompt is built from it."""

    def test_the_acting_tools_are_not_reads(self):
        self.assertEqual(base.REGISTRY["run_command"].effect, "execute")
        self.assertEqual(base.REGISTRY["write_file"].effect, "write")

    def test_the_looking_tools_are_reads(self):
        for name in ("web_search", "fetch_url", "read_file", "list_dir"):
            self.assertEqual(base.REGISTRY[name].effect, "read", name)

    def test_an_effect_reaches_the_rust_side(self):
        self.assertEqual(base.REGISTRY["run_command"].spec()["effect"], "execute")

    def test_a_tool_that_says_nothing_is_unknown_rather_than_read(self):
        @base.tool(name="_silent_for_test")
        def _silent() -> str:
            return ""

        try:
            self.assertEqual(base.REGISTRY["_silent_for_test"].effect, "unknown")
        finally:
            base.REGISTRY.pop("_silent_for_test", None)

    def test_a_misspelled_effect_is_refused_at_definition(self):
        # Silently becoming "unknown" would look like a working declaration.
        with self.assertRaises(ValueError):
            @base.tool(name="_typo_for_test", effect="excute")
            def _typo() -> str:
                return ""
        base.REGISTRY.pop("_typo_for_test", None)


if __name__ == "__main__":
    unittest.main()
