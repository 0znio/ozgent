"""What a command, a fetch and a file tool can reach once they run.

The permission tests decide whether a tool runs. These check what it can then
do — the part an allowlisted ``git`` or ``python``, or a page steering
``fetch_url``, would try next. Each attack here is one a model could be talked
into by text it read.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ozgent_tools import base, http, sandbox
from ozgent_tools.builtin import run_command as _registers  # noqa: F401  (registers the tool)
from ozgent_tools.base import ToolError
from ozgent_tools.permissions import SECTION, approving, private_allowed, resolve_within

HAS_LANDLOCK = sandbox.landlock_abi() > 0


def call(name, **kwargs):
    return asyncio.run(base.REGISTRY[name].fn(**kwargs))


class AddressesTest(unittest.TestCase):
    def test_only_public_addresses_are_public(self):
        private = [
            "127.0.0.1", "10.0.0.1", "172.16.5.4", "192.168.1.1", "169.254.169.254", "100.64.0.1",
            "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fd12::1", "::ffff:127.0.0.1",
            "::ffff:10.0.0.1", "2002:7f00:1::", "64:ff9b::a00:1", "198.51.100.7",
        ]
        for ip in private:
            self.assertFalse(http.is_public(ip), ip)
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8"]:
            self.assertTrue(http.is_public(ip), ip)

    def test_urls_to_this_machine_are_refused(self):
        for url in [
            "http://127.0.0.1:7333/api/settings",
            "http://localhost/",
            "http://[::1]:7333/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::ffff:192.168.1.1]/",
        ]:
            with self.assertRaises(http.HttpError, msg=url):
                asyncio.run(http.check_url(url))

    def test_other_schemes_are_refused(self):
        for url in ["file:///etc/passwd", "gopher://x/", "ftp://example.com/"]:
            with self.assertRaises(http.HttpError, msg=url):
                asyncio.run(http.check_url(url))


class PrivateHostsTest(unittest.TestCase):
    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)

    def test_a_listed_host_may_be_private_and_nothing_else(self):
        base.SHARED_CONFIG[SECTION] = {"network_allow": ["nas.lan"]}
        self.assertTrue(private_allowed("nas.lan"))
        self.assertTrue(private_allowed("files.nas.lan"))
        self.assertFalse(private_allowed("localhost"))
        base.SHARED_CONFIG[SECTION] = {"network_private": True}
        self.assertTrue(private_allowed("localhost"))


class CredentialsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.home = Path(self.dir.name).resolve()
        (self.home / ".ssh").mkdir()
        (self.home / ".ssh" / "id_ed25519").write_text("PRIVATE")
        self.old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)
        base.SHARED_CONFIG[SECTION] = {"root": str(self.home)}

    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)
        if self.old_home is not None:
            os.environ["HOME"] = self.old_home
        self.dir.cleanup()

    def test_a_key_is_refused_even_with_approval(self):
        with approving(True):
            with self.assertRaises(ToolError) as e:
                resolve_within(".ssh/id_ed25519", "read_file")
        self.assertIn("credentials", str(e.exception))

    def test_the_operator_can_allow_them(self):
        base.SHARED_CONFIG[SECTION]["allow_sensitive"] = True
        self.assertEqual(resolve_within(".ssh/id_ed25519", "read_file").name, "id_ed25519")


@unittest.skipUnless(HAS_LANDLOCK, "this kernel has no Landlock")
class CommandSandboxTest(unittest.TestCase):
    """Run real commands through `run_command` and try to get out."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve() / "project"
        self.root.mkdir()
        (self.root / "in.txt").write_text("inside\n")
        # A stand-in for ozgent's own home, beside the project.
        self.protected = Path(self.dir.name).resolve() / "ozgent"
        (self.protected / "configs").mkdir(parents=True)
        (self.protected / "configs" / "config.toml").write_text("api_key = 'secret'\n")
        os.environ["OZGENT_PROTECTED"] = str(self.protected)
        base.SHARED_CONFIG[SECTION] = {
            "root": str(self.root),
            "shell": True,
            "shell_allow": ["cat", "touch", "sh", "env", "ls", sys.executable.rsplit("/", 1)[-1]],
        }
        os.environ["OZGENT_TEST_API_KEY"] = "must-not-leak"

    def tearDown(self):
        base.SHARED_CONFIG.pop(SECTION, None)
        os.environ.pop("OZGENT_PROTECTED", None)
        os.environ.pop("OZGENT_TEST_API_KEY", None)
        self.dir.cleanup()

    def run_(self, command):
        return call("run_command", command=command)

    def test_the_project_is_readable_and_writable(self):
        self.assertIn("inside", self.run_("cat in.txt")["stdout"])
        self.assertEqual(self.run_("touch made.txt")["exit_code"], 0)
        self.assertTrue((self.root / "made.txt").exists())

    def test_ozgent_home_is_unreadable(self):
        # The argument check refuses the real path before anything runs...
        with self.assertRaises(ToolError):
            self.run_(f"cat {self.protected}/configs/config.toml")
        # ...so the sandbox has to hold when the program finds the path itself.
        target = self.protected / "configs" / "config.toml"
        exe = sys.executable.rsplit("/", 1)[-1]
        with approving(True):
            out = self.run_(f"{exe} -c \"print(open(chr(47).join({str(target).split('/')!r})).read())\"")
        self.assertNotEqual(out["exit_code"], 0)
        self.assertNotIn("secret", out["stdout"])

    def test_nothing_outside_the_project_is_writable(self):
        target = Path(self.dir.name).resolve() / "escaped.txt"
        exe = sys.executable.rsplit("/", 1)[-1]
        with approving(True):
            out = self.run_(f"{exe} -c \"open({str(target)!r},'w').write('x')\"")
        self.assertNotEqual(out["exit_code"], 0)
        self.assertFalse(target.exists())

    def test_there_is_no_network(self):
        exe = sys.executable.rsplit("/", 1)[-1]
        script = "print(__import__('socket').create_connection(('1.1.1.1',443),timeout=3) and 'CONNECTED')"
        out = self.run_(f"{exe} -c \"{script}\"")
        self.assertNotIn("CONNECTED", out["stdout"])
        self.assertNotEqual(out["exit_code"], 0)

    def test_secrets_stay_out_of_the_environment(self):
        out = self.run_("env")
        self.assertNotIn("must-not-leak", out["stdout"])
        self.assertIn("TMPDIR=", out["stdout"])

    def test_other_processes_are_invisible(self):
        out = self.run_("ls /proc")
        pids = [p for p in out["stdout"].split() if p.isdigit()]
        self.assertLessEqual(len(pids), 3, pids)


if __name__ == "__main__":
    unittest.main()
