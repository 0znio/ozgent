"""Run one program confined: its own namespaces, a Landlock ruleset, no secrets.

Invoked as a separate interpreter, never imported by the worker to fork from:

    python -S -E sandbox.py '<json policy>' -- program arg...

Starting fresh matters. Applying a sandbox between ``fork`` and ``exec``
(``preexec_fn``) in the worker, which has threads, can deadlock on a lock some
other thread held at the moment of the fork. A new single-threaded process has
no such locks, and nothing it does can disturb the worker.

What the program gets, in the order it is applied:

1. **Namespaces** (``unshare``): a user namespace mapping this user to itself,
   so the program sees ordinary ids; a mount namespace with a fresh ``/proc``
   that lists only the sandbox's own processes, so it cannot read another
   process's environment or memory map; a PID namespace, so the whole tree
   dies with it and nothing it starts can outlive the timeout; and — unless
   the policy allows the network — a network namespace with no interfaces, so
   no TCP, UDP or DNS leaves it.
2. **Landlock**: read and execute only system directories and the toolchains
   in the policy; write only the tool's folder and a private temporary
   directory. ozgent's own home and credential directories are left out even
   when they sit inside the folder. It may not signal processes outside the
   sandbox or reach their abstract sockets.
3. **Limits**: no core files, and a cap on the size of any file it writes.
4. **Environment**: only an allowlist of variables survive. Nothing named like
   a key, token or password, no agent sockets, no display, no session bus.

Standalone on purpose (standard library only, no package imports): it runs with
``-S -E`` so nothing in the environment can put other code in front of it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import resource
import signal
import struct
import sys

# ------------------------------------------------------------------ syscalls

_libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)

SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

PR_SET_NO_NEW_PRIVS = 38
PR_SET_PDEATHSIG = 1

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000

MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REC = 16384
MS_PRIVATE = 1 << 18

# Filesystem rights, by the ABI that introduced them.
FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13  # ABI 2
FS_TRUNCATE = 1 << 14  # ABI 3
FS_IOCTL_DEV = 1 << 15  # ABI 5

NET_BIND_TCP = 1 << 0  # ABI 4
NET_CONNECT_TCP = 1 << 1

SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
SCOPE_SIGNAL = 1 << 1

RULE_PATH_BENEATH = 1

#: Rights that only make sense on a file, as opposed to a directory.
FILE_RIGHTS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV


class RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


def _syscall(number: int, *args) -> int:
    ret = _libc.syscall(ctypes.c_long(number), *args)
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return ret


def landlock_abi() -> int:
    """The Landlock ABI the kernel speaks, or 0 without Landlock."""
    try:
        return _syscall(SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    except OSError:
        return 0


def _prctl(option: int, arg: int) -> None:
    if _libc.prctl(option, ctypes.c_ulong(arg), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


# ------------------------------------------------------------------ policy


def _handled_fs(abi: int) -> int:
    rights = (
        FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_READ_DIR | FS_REMOVE_DIR | FS_REMOVE_FILE
        | FS_MAKE_CHAR | FS_MAKE_DIR | FS_MAKE_REG | FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_BLOCK
        | FS_MAKE_SYM
    )
    if abi >= 2:
        rights |= FS_REFER
    if abi >= 3:
        rights |= FS_TRUNCATE
    if abi >= 5:
        rights |= FS_IOCTL_DEV
    return rights


def _read_rights() -> int:
    return FS_EXECUTE | FS_READ_FILE | FS_READ_DIR


def _write_rights(abi: int) -> int:
    rights = (
        _read_rights() | FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_DIR | FS_MAKE_REG
        | FS_MAKE_SOCK | FS_MAKE_FIFO | FS_MAKE_SYM
    )
    if abi >= 2:
        rights |= FS_REFER
    if abi >= 3:
        rights |= FS_TRUNCATE
    return rights


def _inside(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def grants_except(root: str, excluded: list[str]) -> list[str]:
    """Paths to grant so that all of `root` is reachable except `excluded`.

    Landlock only ever adds access, so a directory cannot be granted with a
    hole in it. Instead each entry of `root` is granted separately, and an
    entry that *contains* an excluded path is descended into rather than
    granted — which leaves the excluded path, and nothing else, unreachable.
    The one thing lost is creating new entries directly in a directory that
    had to be split; everything below it works as before.
    """
    root = os.path.realpath(root)
    holes = [os.path.realpath(e) for e in excluded]
    holes = [h for h in holes if _inside(h, root)]
    if not holes:
        return [root]
    if any(h == root for h in holes):
        return []
    out: list[str] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    for name in entries:
        child = os.path.join(root, name)
        real = os.path.realpath(child)
        if any(h == real for h in holes):
            continue
        if any(_inside(h, real) for h in holes) and os.path.isdir(real):
            out.extend(grants_except(real, holes))
        else:
            out.append(child)
    return out


def _add_path(ruleset: int, path: str, rights: int) -> None:
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return  # a path that does not exist grants nothing
    try:
        if not os.path.isdir(path):
            rights &= FILE_RIGHTS
        # `struct landlock_path_beneath_attr` is packed: a u64 then an s32,
        # twelve bytes with no padding. Laid out by hand rather than with a
        # packed ctypes Structure, whose layout rules differ across versions.
        attr = ctypes.create_string_buffer(struct.pack("<Qi", rights, fd), 12)
        _syscall(SYS_LANDLOCK_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(RULE_PATH_BENEATH),
                 attr, ctypes.c_uint32(0))
    except OSError:
        pass  # e.g. a right this filesystem cannot carry; the path stays shut
    finally:
        os.close(fd)


def apply_landlock(policy: dict, abi: int) -> None:
    handled_fs = _handled_fs(abi)
    handled_net = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 and not policy.get("network") else 0
    scoped = (SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL) if abi >= 6 else 0
    attr = RulesetAttr(handled_access_fs=handled_fs, handled_access_net=handled_net, scoped=scoped)
    size = 8 if abi < 4 else (16 if abi < 6 else 24)
    ruleset = _syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    try:
        excluded = policy.get("protected", [])
        for path in policy.get("read", []):
            for granted in grants_except(path, excluded):
                _add_path(ruleset, granted, _read_rights() & handled_fs)
        for path in policy.get("write", []):
            for granted in grants_except(path, excluded):
                _add_path(ruleset, granted, _write_rights(abi) & handled_fs)
        for path in policy.get("devices", []):
            _add_path(ruleset, path, (FS_READ_FILE | FS_WRITE_FILE) & handled_fs)
        _prctl(PR_SET_NO_NEW_PRIVS, 1)
        _syscall(SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset), ctypes.c_uint32(0))
    finally:
        os.close(ruleset)


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def enter_namespaces(network: bool) -> bool:
    """Unshare what we can. Returns whether a PID namespace was entered."""
    uid, gid = os.getuid(), os.getgid()
    flags = CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | (0 if network else CLONE_NEWNET)
    if _libc.unshare(ctypes.c_int(flags)) != 0:
        return False
    try:
        _write("/proc/self/setgroups", "deny")
        _write("/proc/self/uid_map", f"{uid} {uid} 1")
        _write("/proc/self/gid_map", f"{gid} {gid} 1")
    except OSError:
        pass
    return True


def remount_proc() -> None:
    """A /proc that shows only this PID namespace. Needs the mount namespace."""
    _libc.mount(None, b"/", None, ctypes.c_ulong(MS_REC | MS_PRIVATE), None)
    _libc.mount(b"proc", b"/proc", b"proc", ctypes.c_ulong(MS_NOSUID | MS_NODEV | MS_NOEXEC), None)


def limits(policy: dict) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cap = int(policy.get("max_file_bytes", 2 << 30))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard == resource.RLIM_INFINITY or hard > cap:
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))


def main() -> int:
    try:
        sep = sys.argv.index("--")
    except ValueError:
        print("usage: sandbox.py POLICY -- PROGRAM ARGS...", file=sys.stderr)
        return 126
    policy = json.loads(sys.argv[1])
    argv = sys.argv[sep + 1 :]
    if not argv:
        return 126
    env = policy.get("env", {})
    abi = landlock_abi()
    if abi <= 0 and policy.get("require", True):
        print("ozgent sandbox: this kernel has no Landlock, so the command was not run", file=sys.stderr)
        return 126

    pid_ns = enter_namespaces(bool(policy.get("network")))
    if pid_ns:
        # The next child is PID 1 of the new namespace; when it ends, the
        # kernel ends everything it started.
        child = os.fork()
        if child:
            # Stop the program with us: a timeout kills this process.
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, lambda *_: os.kill(child, signal.SIGKILL))
            _, status = os.waitpid(child, 0)
            return os.waitstatus_to_exitcode(status) if status else 0
        try:
            _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        except OSError:
            pass
        try:
            remount_proc()
        except OSError:
            pass

    try:
        limits(policy)
        if abi > 0:
            apply_landlock(policy, abi)
        os.chdir(policy.get("cwd", "/"))
        os.execvpe(argv[0], argv, env)
    except FileNotFoundError:
        print(f"ozgent sandbox: {argv[0]!r} is not installed or not on PATH", file=sys.stderr)
        return 127
    except OSError as exc:
        print(f"ozgent sandbox: cannot run {argv[0]!r}: {exc}", file=sys.stderr)
        return 126
    return 0


if __name__ == "__main__":
    code = main()
    # Exit codes above 255 are not a thing; negative means a signal.
    sys.exit(code if 0 <= code < 256 else 128 + (-code if code < 0 else 1))
