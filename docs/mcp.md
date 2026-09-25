# MCP servers

The Model Context Protocol is a way for a program to offer tools to a model.
Add a server and its tools appear alongside ozgent's own: same model, same
permission rules, same prompt. They reach every way you talk to ozgent — the
web page, the terminal, Telegram and WhatsApp, scheduled jobs, agents and the
API — because they all go through the one daemon that runs the servers.

```
ozgent mcp                          connect to each one and show what it offers
ozgent mcp search github            find one in the MCP Registry
ozgent mcp install <registry name>  install it
```

## Adding one

### On the admin page

**Admin → MCP → Add a server** has three ways in:

- **MCP Registry.** Search the official registry
  (<https://registry.modelcontextprotocol.io>), pick a server, fill in what it
  asks for (API keys are password fields), and choose **Review**. ozgent shows
  the exact command it will run; **Add server** saves it.
- **npm or PyPI.** A package by name: npm packages run with `npx`, PyPI ones
  with `uvx`.
- **Command or URL.** Any program that speaks MCP over stdio, or a server
  reached over HTTPS.

Nothing is saved until it has been reviewed, and a registry install never
takes a command from the page: the page says *which* registry entry, and
ozgent looks it up again and builds the command itself.

The registry is open — anyone can publish under a namespace they own — so an
entry is someone's claim about their own code. The page links each server's
source repository; read it before installing something that will run on your
machine.

### From the terminal

```bash
ozgent mcp search filesystem
ozgent mcp install io.github.YawLabs/fetch-mcp
ozgent mcp install io.github.example/files --set arg:0=/home/you/notes

ozgent mcp add files --npm @modelcontextprotocol/server-filesystem -- ~/notes
ozgent mcp add time  --pypi mcp-server-time -- --local-timezone Europe/London
ozgent mcp add mine  -- /usr/local/bin/my-server --flag
ozgent mcp add docs  --url https://example.com/mcp --header "Authorization=Bearer …"
```

`install` asks for anything the server needs that `--set` did not give,
shows the command, and asks before saving (`--yes` to skip that). A running
daemon notices the change within a few seconds and connects to the new server;
nothing needs restarting.

```bash
ozgent mcp disable files     switch one off, keeping its settings
ozgent mcp enable files
ozgent mcp off               stop using MCP servers at all
ozgent mcp on
ozgent mcp remove files      remove it, and the rules set for its tools
```

### In `config.toml`

```toml
[mcp]
enabled = true

[mcp.servers.files]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]
sandbox = true
folders = ["/home/you/notes"]

[mcp.servers.github]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-github"]
env     = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_…" }

[mcp.servers.docs]
url     = "https://example.test/mcp"
headers = { Authorization = "Bearer …" }
```

A server is one or the other. Giving both, or giving `env` to an HTTP server,
is refused rather than half-applied — silently dropping the `env` that carries
your API key looks exactly like the server rejecting your credentials, and that
is a long afternoon.

## The sandbox

A stdio server is someone else's program running as you. Installed from a
registry, it is code nobody here has read. With `sandbox = true` — the
default for servers added from the admin page or `ozgent mcp` — it runs in the
same sandbox as `run_command`:

- **Its own home.** `HOME`, its caches and the npm and uv package caches point
  at `~/ozgent/mcp/<name>/`, so what it downloads cannot touch the packages
  your own tools run from, and what it finds under `~` is its own.
- **The system, not your files.** It may read the whole system — programs,
  libraries, settings, `/sys` — except the private places: people's home
  folders, `/root`, shared temporary folders, your session's runtime folder,
  mail and mounted drives. It writes only its home and the `folders` you give
  it. A folder named in its own arguments (`server-filesystem ~/notes`)
  counts as given. It may also read, not write, the package it was unpacked
  from when that is under `~/ozgent/mcp` — all of it, not just the folder
  the program sits in, since a built server in `release/` starts the browser
  beside it — and any existing folder its `env` names (`GHOSTFOX_HOME=…`),
  except inside ozgent's own directories or your credential folders.
- **Its own temporary and shared-memory folders,** private to it, where the
  kernel allows namespaces; your session's sockets (D-Bus, the keyring,
  agents) are hidden behind empty private mounts, because Landlock stops
  reading files but not connecting to a socket.
- **Not your secrets.** ozgent's own home, your SSH, cloud and browser
  credentials, and other processes are out of reach, even inside a folder you
  gave it.
- **Only the environment it is given.** ozgent's own keys and tokens are not
  passed on, sandboxed or not; a server gets the basics a program needs
  (`PATH`, `HOME`, locale, proxy settings) and its own `env`.
- **The network if you allow it** (`network = true`, the default). Most
  servers need it, if only for `npx` or `uvx` to fetch the package; switch it
  off for a server that only works on local files.

`ozgent mcp` and the admin page say which servers are sandboxed. Removing a
server deletes its home.

Everything a server keeps is in `~/ozgent/mcp/<name>`: its sandbox home, and
for a server without the sandbox, what `npx` and `uvx` download for it.
Deleting `~/ozgent/mcp` removes every trace. (A temporary folder goes under
the system's own only when that path would be too long for a socket, and
container images live in Docker's storage.)

Why "everything but the private places" rather than a list of what may be
read: programs look in more places than any list anticipates. Chromium probes
`/sys` and needs shared memory; on Ubuntu, DNS goes through
`/etc/resolv.conf` into `/run`; tools install into `/usr/local`, `/var/lib`
and `/opt`. Each of those broke an allowlist in turn. What needs protecting is
short and known — your files, your secrets, your session — so that is what is
shut.

If a sandboxed server fails, its card says so and suggests what to try: give
it the folder it needs, or switch the sandbox off once to see whether it is
the cause.

A server whose sandbox cannot be set up (no Landlock in the kernel, or
ozgent's Python runtime not found) is not started, and says why. On systems
without unprivileged user namespaces — stock Ubuntu 24.04 — the sandbox still
applies, as described in [tools.md](tools.md#what-the-boundaries-actually-guarantee).

## Choosing its tools, and what they may do

Each server's card on the admin page lists every tool it offers. Untick a tool
to keep it from the model, and set what happens when the model calls one:

| | |
|---|---|
| default | the rule for its kind: every MCP tool asks, unless the server is trusted |
| runs without asking | `allow` |
| asks first | `ask` |
| never runs | `deny` |

**All its tools** on the card sets one rule for every tool of the server at
once — a browser server with fifty tools is one decision, not fifty. A rule
set on a single tool still wins over it, so a server can run without asking
except for the one tool that deletes things. When a server's tool asks, the
prompt in the chat offers **always, all github tools** beside **always**.

The same in `config.toml`:

```toml
[mcp.servers.github]
tools = ["search_issues", "get_issue"]   # by the tool's own name on the server

[permissions.tools]
"github_*" = "allow"                      # every github tool
github_delete_file = "ask"                # except this one
github_search_issues = "allow"            # by the name the model sees
```

A `*` at the end covers every tool whose name begins that way; the longest
such rule that matches wins, and a tool's own rule beats all of them.

## Choosing which servers your chats use

The admin decides which servers exist and run. Whoever chats chooses which of
them to use, the way single tools are chosen:

- **Settings → Tools → MCP servers** on the chat page: untick a server to
  leave it out of the chats. It keeps running, for the other chats' channels
  and anyone who has it on; the model is simply not offered its tools. Single
  tools, a server's included, are unticked in the list above it.
- **`/mcp off <server>`** in the terminal, and `/mcp on <server>` to use it
  again; `/mcp off` alone leaves them all out.
- The **MCP** button beside **Web** and **Tools** under the message box opens
  a list of the running servers, one switch each, with their tools folded
  underneath. Like the Tools tray it is remembered by that browser for every
  conversation until you change it, and it takes effect from the next
  message. Servers that are switched off, failed or still connecting are
  listed too, with the reason.

These are stored as `[tools] mcp_off` and `[tools] disabled`, need no admin
password and no restart, and apply from the next message.

## Names

A server's tools are prefixed with the server's name: `files_read_file`.
Always, not only when two servers collide.

That is worth a sentence, because the alternative is tempting. If names were
only prefixed on collision, adding an unrelated server would silently rename an
existing tool — invalidating any `[permissions.tools]` rule written about it,
and changing the tool list mid-conversation for a model that had already been
told what it could call.

If two servers do offer the same prefixed name, the first one keeps it and the
other is reported by `ozgent tools` rather than silently dropped. ozgent's own
Python tools always win, so a server cannot take a name out from under
`write_file`.

## Why every MCP tool asks by default

MCP tools can carry annotations — `readOnlyHint` and friends. ozgent ignores
them unless you say otherwise, and every tool from a server is treated as
having declared nothing, which means [it asks](tools.md).

That is not excess caution. `readOnlyHint` decides whether a call runs *without
anyone being asked*, and it is written by the same party that wants the call to
happen. The specification says as much: annotations are hints, and a client
must not trust them unless the server is trusted. ozgent's own rule already
says a tool that does not declare its effect gets asked about, because silence
must not be read as harmless — and a claim about itself is not better evidence
than silence.

For a server you run yourself, say so — **Trust** on its card, or:

```toml
[mcp.servers.files]
trust_hints = true
```

Then a tool the server calls read-only runs under your `read` rule, which
normally means without asking; anything else still asks. A tool that says
nothing either way still asks, even from a trusted server — trusting a server
means believing what it says, not filling in what it did not.

## In the terminal, on your phone, and on a timer

The terminal is a client of the daemon, so it has the daemon's servers. `/mcp`
lists them and how they are doing, `/mcp files` lists one server's tools, and
`/mcp off files` or `/mcp on files` chooses whether your chats use it. A tool
that asks shows the usual permission prompt.

Telegram and WhatsApp chats get the servers' tools too, limited by the
channel's own tool list; whether someone there may approve a tool that asks is
the channel's **Approvals** setting. A scheduled job has nobody to ask, so a
tool that asks is refused there: allow the ones a job needs.

## Over the API

MCP tools are part of ozgent's tools, so an API caller uses them the same way:

```bash
curl http://127.0.0.1:7333/v1/chat/completions -d '{
  "model": "Qwen3.5-4B:Q4_K_M",
  "messages": [{"role": "user", "content": "What time is it in Tokyo?"}],
  "ozgent_tools": true,
  "native_tools": ["time_get_current_time"]
}'
```

A program cannot answer a permission prompt, so a tool that asks is refused
for an API caller. Allow the tools a program should use. An API key also
needs the scopes for them: an MCP tool that does not say what it does counts
as writing *and* running programs, so the key needs both of those scopes —
or trust the server, and its read-only tools need only the read-only tools
scope.

`GET /api/mcp` lists the servers and how they are doing, for the web page and
the terminal; the settings themselves are only on the admin routes (see
[api.md](api.md)).

## Many servers, many tools

Every tool described to the model costs context before anyone has said
anything: six servers brought 119 tools and 38,896 tokens of descriptions. So
once the tools together pass about 4,000 tokens, a server's tools are no longer
described up front. The model is given a directory of them instead — names
and what they do, as much as fits — and, with each message, the full
descriptions of the few that message is most likely about. It can look up
others with `find_tools`, and calls any of them by name.

Nothing needs setting for this, and it scales: on the MCP-Zero set of 2,797
tools from 308 servers, a fitting tool was among the ten put in front of the
model for 89% of requests, against 92% with 130 tools. How it works, and what
was measured, is in [tools.md](tools.md#hundreds-of-tools-or-thousands).

A server can say otherwise, with **Tools up front** on its card or `load`:

```toml
[mcp.servers.files]
load = "always"      # always described up front, whatever the count
# load = "on_request"  # never up front, even when there is room
# load = "auto"        # the default: up front while everything fits
```

`always` is for a server used in most conversations, whose tools the model
should never have to look up. ozgent's own tools are always up front.

## Everything else

```toml
[mcp.servers.<name>]
enabled         = true
description     = "What it is, for lists"
timeout_seconds = 60      # one call; MCP servers are often network clients
network         = true    # with sandbox = true
```

A server that will not start is reported and skipped. One broken entry never
takes away the tools that work, and never stops ozgent starting. Servers start
at the same time, so one that takes a minute to download does not hold up the
rest.

## What is not supported

- **Resources and prompts.** ozgent uses `tools/list` and `tools/call` only.
- **Servers calling back.** `sampling` and `elicitation` let a server ask the
  client to run a model or ask the user a question. ozgent advertises neither,
  so no server will try — which is deliberate: a tool that can silently spend
  your tokens or put its own words in a prompt is a different trust decision
  from a tool that returns data.
- **OAuth.** HTTP servers are authenticated with whatever you put in `headers`.
- **The old SSE transport**, and packages that start their own web server
  on a port. The registry lists which is which; the admin page greys out the
  ones ozgent cannot run and says why.
- **Container images** are installable when Docker is installed; they run
  in Docker, not in ozgent's sandbox.

## When something is wrong

**A server is listed as failed.** Its card on the admin page, and `ozgent mcp`,
show the error and the last lines it wrote. A server that exits at once is
reported at once — commonly a package that does not exist, or a missing API
key.

**"could not start".** The `command` is not on ozgent's `PATH`. npm servers
need Node.js (`npx`), PyPI servers need `uv` (`uvx`), and the first run
downloads the package, which is why the handshake is allowed two minutes.

**A sandboxed server cannot find a file.** Give it the folder: **Folders it may
use** on its card, `--folder` with `ozgent mcp add`, or `folders = [...]`.

**A tool always asks even though it is read-only.** That is the default; set
**Trust** on that server, or allow the tool.

**Searching the registry is slow.** Its search sometimes takes twenty seconds
or more, and occasionally times out for a particular word; try a shorter or
different one.

See also [tools and permissions](tools.md) and [settings](settings.md).
