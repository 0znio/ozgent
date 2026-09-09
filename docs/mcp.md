# MCP servers

The Model Context Protocol is a way for a program to offer tools to a model.
Point ozgent at a server and its tools appear alongside ozgent's own — same
model, same permission rules, same prompt.

```
ozgent mcp        connect to each one and show what it offers
```

## Adding one

Most servers are a program to run. In `~/ozgent/configs/config.toml`:

```toml
[mcp]
enabled = true

[mcp.servers.files]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]

[mcp.servers.github]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-github"]
env     = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_…" }
```

Servers reached over HTTP take a `url` instead:

```toml
[mcp.servers.docs]
url     = "https://example.test/mcp"
headers = { Authorization = "Bearer …" }
```

A server is one or the other. Giving both, or giving `env` to an HTTP server,
is refused rather than half-applied — silently dropping the `env` that carries
your API key looks exactly like the server rejecting your credentials, and that
is a long afternoon.

Then check it:

```
$ ozgent mcp
files
  hints         not trusted — every tool asks
  tools         4
    files_read_file                    unknown  ask
      Read the complete contents of a file.
    files_write_file                   unknown  ask
      Create a new file or overwrite an existing one.
```

## Names

A server's tools are prefixed with the server's name: `files_read_file`. Always,
not only when two servers collide.

That is worth a sentence, because the alternative is tempting. If names were
only prefixed on collision, adding an unrelated server would silently rename an
existing tool — invalidating any `[permissions.tools]` rule written about it,
and changing the tool list mid-conversation for a model that had already been
told what it could call.

If two servers do offer the same prefixed name, the first one keeps it and the
other is reported by `ozgent tools` rather than silently dropped. ozgent's own
Python tools always win, so a server cannot take a name out from under
`write_file`.

## Permission, and why every MCP tool asks by default

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

For a server you run yourself, say so:

```toml
[mcp.servers.files]
trust_hints = true
```

Then a tool the server calls read-only runs under your `read` rule, which
normally means without asking; anything else still asks. A tool that says
nothing either way still asks, even from a trusted server — trusting a server
means believing what it says, not filling in what it did not.

You can also decide per tool, exactly as for ozgent's own:

```toml
[permissions.tools]
files_read_file  = "allow"
files_write_file = "deny"
```

## Narrowing what a server offers

```toml
[mcp.servers.github]
tools = ["search_issues", "get_issue"]
```

Named by the tool's own name on the server, not the prefixed one. Everything
else it lists is not offered to the model at all.

## Everything else

```toml
[mcp.servers.<name>]
enabled         = true
timeout_seconds = 60      # one call; MCP servers are often network clients
```

A server that will not start is reported and skipped. One broken entry never
takes away the tools that work, and never stops ozgent starting.

## What is not supported

- **Resources and prompts.** ozgent uses `tools/list` and `tools/call` only.
- **Servers calling back.** `sampling` and `elicitation` let a server ask the
  client to run a model or ask the user a question. ozgent advertises neither,
  so no server will try — which is deliberate: a tool that can silently spend
  your tokens or put its own words in a prompt is a different trust decision
  from a tool that returns data.
- **OAuth.** HTTP servers are authenticated with whatever you put in `headers`.

## When something is wrong

**`ozgent mcp` shows nothing.** `[mcp] enabled` defaults to false.

**A server is listed but has no tools.** It connected and said it offers none —
it may only provide resources or prompts, which ozgent does not use.

**"could not start".** The `command` is not on ozgent's `PATH`. `npx`-based
servers need Node installed, and the first run downloads the package, which is
why the handshake is allowed two minutes.

**A tool always asks even though it is read-only.** That is the default; set
`trust_hints = true` on that server, or allow the tool by name.

See also [tools and permissions](tools.md) and [settings](settings.md).
