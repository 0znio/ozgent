# Tools and permissions

ozgent ships seven tools. Three of them can change something or reach outside
the machine, and those are **off until you turn them on**.

The model decides when to use a tool; you find out afterwards. So the ones that
only read something you already pointed at need no ceremony, and the ones that
write, execute, or fetch are deny-by-default. Every refusal names the exact
setting that would permit it — a permission system nobody can work out how to
grant is one people disable wholesale.

| tool | what it does | permission |
|---|---|---|
| `read_file` | read a file, or the parts matching a query | confined to `root` |
| `list_dir` | list a directory, up to 4 levels | confined to `root` |
| `web_search` | search the web | provider key |
| `write_file` | create or modify a file | **`write = true`** |
| `run_command` | run one allowed program | **`shell = true`** + allowlist |
| `fetch_url` | read a web page | **`network = true`** |
| `get_temperature` | example tool | — |

## Where they live

One tool per file, and the file is named after the tool. The six built-ins are
in `python/ozgent_tools/builtin/` — `read_file.py`, `list_dir.py`,
`web_search.py`, `write_file.py`, `run_command.py`, `fetch_url.py`. The
seventh, `get_temperature`, is the example in `~/ozgent/tools/`, which is where
your own go.

`ozgent tools list` prints the file each tool came from, so the two are never
in doubt:

```
$ ozgent tools list
7 tools · python 3.14.6 · worker 0.1.0

read_file
  ~/ozgent/lib/python/ozgent_tools/builtin/read_file.py
  Read a file, or the parts of it relevant to a query.
...
get_temperature
  ~/ozgent/tools/demo.py
  Get the current temperature for a city.
```

Adding one is adding a file to `~/ozgent/tools/`; see [Adding your own tool](#adding-your-own-tool).

## Configuring

All of it lives in `~/ozgent/configs/config.toml`:

```toml
[tools.config.permissions]
root          = "/home/you/code"   # filesystem boundary for every file tool
write         = false              # create or modify files
shell         = false              # run commands
shell_allow   = ["git", "cargo"]   # and only these programs
network       = true               # fetch pages by URL
network_allow = []                 # empty means any host
```

`root` defaults to the directory ozgent was started in — the least surprising
boundary, since a model asked about "this project" should not reach the rest of
the disk. A single tool can override it:

```toml
[tools.config.read_file]
root = "/home/you/notes"           # this tool only
```

## What the boundaries actually guarantee

**Paths are resolved before they are checked.** `../../../etc/passwd` and a
symlink pointing out of the tree both fail, because the check runs on the
resolved path rather than the string.

**`shell = true` on its own permits nothing.** The allowlist starts empty and
matches on the *program*, not the whole command line, so `shell_allow = ["ls"]`
permits `ls -la` and `/bin/ls`, and refuses `rm` and `/bin/rm`. Enumerating
what is safe is possible; enumerating what is dangerous is not.

**Commands run without a shell.** Pipes, redirection, `&&`, backticks and `$()`
have no effect, so they are refused outright rather than silently meaning
something else — otherwise an allowed program could introduce a disallowed one.

**`network_allow` matches hosts and their subdomains.** `["example.com"]`
permits `example.com` and `docs.example.com`, and refuses `notexample.com`.
Only `http` and `https` are fetchable; `file://` is not.

These are the properties the tests in `python/tests/test_permissions.py` pin
down. They are the tests worth reading before changing any of this: everything
else in the tool layer fails visibly, while a permission check that is subtly
wrong fails by letting the model do something you never agreed to.

## Adding your own tool

One file per tool in `~/ozgent/tools/`, discovered on startup:

```python
from typing import Annotated
from ozgent_tools.base import tool

@tool
async def get_temperature(
    city: Annotated[str, "City name, e.g. 'Oslo'."],
) -> dict:
    """Get the current temperature for a city."""
    return {"city": city, "celsius": 12}
```

The first line of the docstring is what the model sees, so it has to carry the
whole rule about when to use the tool. Annotations become the JSON Schema the
model is constrained to. A tool that needs a boundary should use
`ozgent_tools.permissions` rather than rolling its own:

```python
from ozgent_tools.permissions import resolve_within, require

path = resolve_within(user_supplied, "my_tool")   # confined to root
require("write", "modifying the index")           # deny-by-default
```

Inspect what loaded with `ozgent tools list`.

## In the API

Tools are off for API requests unless asked for:

```json
{ "ozgent_tools": true }
```

Restrict which ones the model may see:

```json
{ "ozgent_tools": true, "native_tools": ["list_dir", "read_file"] }
```

Anything not named is withheld — the model cannot call it even if it wants to.
That is a second, per-request layer on top of the config above; a tool must
pass both.

## The agent loop

A turn may call tools up to **eight** times before it must answer. Each round
is a full generation, so the number is a latency budget as much as a capability
one. When the rounds run out the tools are taken away and the model is asked to
answer from what it has — and to say what is missing rather than invent it.

A turn that ends without answering is asked once more. A model can close its
reasoning and stop without either answering or calling anything, and the reply
would otherwise be an empty string.
