# Tools and permissions

ozgent ships seven tools. The model decides when to use one; you find out
afterwards. Two things stand between the decision and the machine, and they
answer different questions.

**Does this call happen at all?** ozgent asks you, at the moment of the call,
showing the tool and its arguments. Asking about everything trains people to
hit yes without reading, so the question is asked once per *kind* of thing: a
tool declares what it does, and the policy answers per kind.

| the tool | out of the box |
|---|---|
| reads something | runs, no question |
| changes files or data | asks |
| runs a program | asks |
| does not say what it does | asks |

Answer once with "always allow" and it becomes a per-tool rule you can change
later — the same rule `/permissions`, the Settings page and `config.toml` all
edit. See [Deciding what runs](#deciding-what-runs).

**What may it touch once it is running?** A root directory for the file tools,
an allowlist for the shell, one for the network. This is the older layer and it
is configured ahead of time, in `[tools.config.permissions]`. A call *you*
approved is past the question this layer exists to ask, so approving lifts it
for that call — refusing to run a command you have just read and approved,
because a flag you have never seen is off, is a permission system arguing with
its own user.

| tool | what it does | declares | boundary |
|---|---|---|---|
| `read_file` | read a file, or the parts matching a query | read | confined to `root` |
| `list_dir` | list a directory, up to 4 levels | read | confined to `root` |
| `web_search` | search the web | read | provider key |
| `fetch_url` | read a page, article, JSON API, or feed | read | `network` + host allowlist |
| `write_file` | create or modify a file | write | `write = true`, confined to `root` |
| `run_command` | run one allowed program | execute | `shell = true` + allowlist |
| `yahoo_finance` | quotes, price history, technical indicators, fundamentals, news, symbol search | read | — |
| `reddit` | search posts, list a subreddit, read a thread | read | optional app credentials |

**`yahoo_finance`** needs no key. One tool with an `action` — `quote`,
`history`, `technicals`, `fundamentals`, `news`, `search` — because a small
model picks the right action from one description far more reliably than the
right tool out of six. Every number comes back raw, never as Yahoo's `"3.2T"`.

`technicals` computes, from two years of daily prices: 20/50/200-day and
12/26 EMA averages, RSI(14), MACD(12,26,9), Bollinger bands, ATR, support and
resistance from recent swing points, the 52-week range, returns over a week to
a year, and the volume trend. Textbook definitions (Wilder's smoothing for RSI
and ATR), so every number can be checked against a charting site — plus
plain-words `signals` ("RSI 74: overbought by the usual 70 rule") that a small
model would otherwise misread from the raw numbers.

**`reddit`** works without setup, but slowly: Reddit refuses anonymous API
calls, so it reads the public feeds, which allow about one request a minute
and carry no scores. For the real thing, create a free *script* app at
<https://www.reddit.com/prefs/apps> and add it:

```toml
[tools.config.reddit]
client_id = "..."          # or $REDDIT_CLIENT_ID
client_secret = "..."      # or $REDDIT_CLIENT_SECRET
username = "your_name"     # Reddit asks apps to name who runs them
```

Every result says which route it came from, and a spent rate limit is
reported with how long until the next request rather than retried into a
longer one.

**Choosing tools per chat.** In the web interface the **Web** switch next to
the message box turns searching and page-reading on and off, and **Tools**
opens a tray with a switch for every other tool — remembered by that browser.
They decide what the model may reach for; an agent you call by name keeps its
own tools.

Tools can also come from [MCP servers](mcp.md). They go through everything
below unchanged, with one difference stated there: a server's claim that a tool
is read-only is not believed unless you mark that server trusted, so out of the
box every tool from a server asks.

## Deciding what runs

When a tool asks, the terminal shows the question on the bar above the status
line and the web interface shows it as a card in the conversation. Four
answers, everywhere:

```
▶ run_command  git status     1 yes · 2 session · 3 always · 4 no
```

`session` lasts until ozgent exits and is never written to disk. `always`
writes a rule:

```toml
[permissions]
read    = "allow"   # tools that only look something up
write   = "ask"     # tools that change files or data
execute = "ask"     # tools that run programs
unknown = "ask"     # tools that do not declare an effect

[permissions.tools]
run_command = "allow"   # a named tool beats its kind
write_file  = "deny"
```

From the terminal:

```
/permissions                       what every tool may do, and why
/permissions run_command allow     a rule for one tool
/permissions run_command clear     back to its kind
/permissions execute deny          a rule for a whole kind
```

or from Settings → Permissions in the web interface. All three read and write
the same file.

Two cases have nobody to ask, and both refuse: a chat driven from a pipe, and
the OpenAI-compatible API, where the caller is a program and cannot consent on
a person's behalf. An operator who wants those tools available there says so in
the policy above.

A tool declares its own effect, because ozgent cannot know what your Python
does — see [Adding your own tool](#adding-your-own-tool). A tool that declares
nothing is asked about, which is what silence has to mean.

## Where they live

One tool per file, and the file is named after the tool. The eight built-ins
are in `python/ozgent_tools/builtin/` — `read_file.py`, `list_dir.py`,
`web_search.py`, `write_file.py`, `run_command.py`, `fetch_url.py`,
`yahoo_finance.py`, `reddit.py`. Your own go in `~/ozgent/tools/`; the
`get_temperature` below is the kind of thing that lives there.

`ozgent tools list` prints the file each tool came from, so the two are never
in doubt:

```
$ ozgent tools list
9 tools · python 3.14.6 · worker 0.1.0

read_file
  ~/ozgent/lib/python/ozgent_tools/builtin/read_file.py
  Read a file, or the parts of it relevant to a query.
...
get_temperature
  ~/ozgent/tools/demo.py
  Get the current temperature for a city.
```

Adding one is adding a file to `~/ozgent/tools/`; see [Adding your own tool](#adding-your-own-tool).

## Configuring the boundaries

The second layer — what a tool may touch once it is running — lives in
`~/ozgent/configs/config.toml`, alongside the `[permissions]` policy above.
Keep the two apart in your head: `[permissions]` decides whether a call
happens, and this decides how far it can reach.

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

## What `fetch_url` returns

Not the page. The *article* in the page.

A stripped-tags dump of any real URL is a navigation menu — Wikipedia opens
with sixty lines of it — so a model handed the first 12,000 characters gets a
table of contents and no story. That fetch succeeds and the answer is useless,
which is worse than failing. Three strategies run, best first:

| `strategy` | when | what it means |
|---|---|---|
| `json-ld` | news sites | the publisher's own `articleBody`, already clean |
| `readability` | most pages | the block with the most text and the fewest links |
| `whole-page` | short or unusual pages | tags stripped, everything kept |

Content type decides the rest: JSON is pretty-printed, RSS and Atom give up
their entries, `text/*` is passed through untouched, PDFs are read if `pypdf`
is installed and refused clearly if not, and anything binary is refused by
name rather than dumped as noise.

Pass `query` on a long page and it comes back as the paragraphs that match,
in their original order, instead of the first few thousand characters.

Two things it deliberately does not do. It does not run JavaScript — a page
built entirely in the browser comes back with a note saying so rather than an
empty string. And it sends a browser `User-Agent`, because a great many sites
answer an unrecognised one with 401 regardless of `robots.txt`; the request is
still one page, on demand, because you asked for it. Sites that block harder
than that (Reuters, at the time of writing) are reported as blocking, with the
status code, rather than as a vague failure.

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

@tool(effect="read")
async def get_temperature(
    city: Annotated[str, "City name, e.g. 'Oslo'."],
) -> dict:
    """Get the current temperature for a city."""
    return {"city": city, "celsius": 12}
```

The first line of the docstring is what the model sees, so it has to carry the
whole rule about when to use the tool. Annotations become the JSON Schema the
model is constrained to.

`effect` says what the tool does to the world, and decides whether ozgent runs
it without asking: `"read"` looks something up, `"write"` creates or changes
something, `"execute"` runs a program. Left off it is `"unknown"`, and ozgent
asks — the right default for a tool whose author has not thought about it, and
the reason every tool written before this existed still works. Declare it
honestly: a tool that quietly writes files while claiming to read is the one
way to defeat the prompt.

A tool that needs a boundary should use `ozgent_tools.permissions` rather than
rolling its own:

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

A turn may call tools up to **eight** times before it must answer ([agents](agents.md)
set their own number). Each round
is a full generation, so the number is a latency budget as much as a capability
one. When the rounds run out the tools are taken away and the model is asked to
answer from what it has — and to say what is missing rather than invent it.

A turn that ends without answering is asked once more. A model can close its
reasoning and stop without either answering or calling anything, and the reply
would otherwise be an empty string.
