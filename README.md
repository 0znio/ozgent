# ozgent

Run language models on your own machine — and let them actually *do* things.

ozgent is one Rust binary over llama.cpp. It gives you a terminal app, a web
interface, an OpenAI-compatible API, tools that really run, and a permission
prompt before anything touches your disk.

![ozgent web interface](docs/images/web.png)

```bash
ozgent                    # chat in the terminal
ozgent web                # chat in a browser
ozgent serve              # OpenAI- and Anthropic-compatible API on :7337
ozgent gateway            # answer messages on Telegram and WhatsApp
```

---

## Why not Ollama or LM Studio?

They run models. ozgent runs models **and does the work around them**.

|  | ozgent | Ollama | LM Studio |
|---|---|---|---|
| Runs tools for you | built in — search, fetch, files, shell, markets, Reddit | you write the client | you wire it up |
| Agents you call with `@name` | yes, with their own tools | — | — |
| **Asks before writing or running** | yes, before the content is even generated | — | — |
| MCP servers | yes | — | yes |
| Full-screen terminal UI | yes | plain prompt | — |
| Web interface | built in | desktop app | desktop app |
| OpenAI-compatible API | yes, and Anthropic-compatible | yes | yes |
| Chat from your phone | Telegram, WhatsApp | — | — |
| Remembers across chats | facts + retrieval | — | — |
| Context sized to your VRAM | worked out for you, and reported | set by hand | set by hand |
| Licence | MIT | MIT | closed source |

*Both projects move fast; check their docs if a row matters to you.*

The one that matters most in practice: **ozgent asks before it acts.** A model
that wants to write a file or run a command stops and shows you what it is
about to do — *before* it spends a minute generating the file contents.

![permission prompt](docs/images/web-permission.png)

---

## Install

```bash
git clone https://github.com/0znio/ozgent
cd ozgent
./install.sh
```

That works out your distribution, installs what the build needs, picks a GPU
backend, compiles, and links `ozgent` onto your `PATH`. It is safe to re-run.

Use `./install.sh --dry-run` first if you want to see what it would do.

| | |
|---|---|
| `--backend cuda\|vulkan\|metal\|cpu` | override the detection |
| `--prefix ~/.local` | install somewhere else |
| `--skip-deps` | don't install system packages |
| `--uninstall` | remove it again |

Debian, Ubuntu, Arch, Fedora, openSUSE, Alpine and macOS are handled; anything
else needs `--skip-deps` and a C++ compiler, CMake, git and Python 3.

<details>
<summary>Or build it yourself</summary>

Needs [Rust](https://rustup.rs) 1.85+, a C++ compiler and CMake.

```bash
cargo build --release --features cuda   # or vulkan, metal, or nothing for CPU
```

The binary is `target/release/ozgent`.
</details>

### On another machine

`./scripts/package.sh` builds a tarball with the binary, the Python tools and
an installer — nothing is compiled on the target. See
[docs/install.md](docs/install.md).

### First run

In the browser: `ozgent web`, then the **download button** next to the model
picker. Search Hugging Face, pick a size (it says which fit your GPU), watch
it download. Or from the terminal:

```bash
ozgent pull unsloth/Qwen3.5-4B-GGUF:Q4_K_M   # any GGUF repo, fetched in parallel
ozgent list                                  # what you have
ozgent default Qwen3.5-4B:Q4_K_M             # use it when none is named
ozgent                                       # chat
```

`ozgent pull <repo> --list` shows a repo's quantisations without downloading.
`ozgent doctor` reports your GPU, backends and anything misconfigured.

---

## What you get

### A terminal app, not a prompt

Full screen, reflows on resize, markdown and syntax highlighting, and a status
bar showing the model, context used, and tokens/sec.

![terminal interface](docs/images/tui.png)

### Tools that run, with a prompt first

Eight built in: fetch a page, read and write files, list directories, run a
command, search the web, live market data, and read Reddit. The model
decides; you approve.

```
✎ write_file  haiku.txt   1 yes · 2 session · 3 always · 4 no
```

![permission prompt in the terminal](docs/images/tui-permission.png)

Reads run without asking. Writes and commands ask. Answer *always* once and it
becomes a rule you can edit later.

Adding your own is a Python file in `~/ozgent/tools`:

```python
from typing import Annotated
from ozgent_tools import tool

@tool(effect="read")
async def define(word: Annotated[str, "The word to look up."]) -> dict:
    """Look up what a word means."""
    return {"word": word, "meaning": "a small furry animal"}
```

The signature becomes the schema the model is shown — there is no manifest to
keep in sync.

→ [docs/tools.md](docs/tools.md)

### Agents

Write `@name` and the message goes to an agent: its own instructions, **only
its own tools**, and its work shown right where it ran.

```
@stock-guru how is NVDA doing after earnings?
@deep-researcher what changed in the EU AI Act this year?
@sentiment-analyser what does Reddit think of the new Pixel?
```

Type `@` for the list. Make your own in Settings → Agents, or
`ozgent agent new <name>`.
→ [docs/agents.md](docs/agents.md)

### MCP servers

Point ozgent at any Model Context Protocol server and its tools join the list.

```toml
[mcp]
enabled = true

[mcp.servers.files]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "~/notes"]
```

Every MCP tool asks before running, because a server's own "this is read-only"
claim is written by the thing that wants to be run.
→ [docs/mcp.md](docs/mcp.md)

### Chat from your phone

Telegram (a bot token) and WhatsApp (link your own account). Nobody can talk to
it until you allow them.

```bash
ozgent channel install whatsapp   # one npm install
ozgent channel login whatsapp     # scan a QR with your phone
ozgent gateway                    # prints a pairing code
```

→ [docs/channels.md](docs/channels.md)

### It remembers

Conversations live in SQLite and are shared by every surface — start in the
terminal, continue in the browser, pick it up on your phone. Facts worth
keeping are retrieved into later chats.

### An OpenAI- and Anthropic-compatible API

```bash
ozgent serve
curl http://127.0.0.1:7337/v1/chat/completions \
  -d '{"model":"Qwen3-4B:Q4_K_M","messages":[{"role":"user","content":"hi"}]}'
curl http://127.0.0.1:7337/v1/messages \
  -d '{"model":"Qwen3-4B:Q4_K_M","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}'
```

Point any OpenAI or Anthropic client at `http://127.0.0.1:7337/v1`. Streaming,
tool calls, reasoning and embeddings all work, and so do agents — `@name` in a
message, or picked as the model. `ozgent web` serves the same API on its own
port, over the model it already has loaded.
→ [docs/api.md](docs/api.md)

### Context sized to your hardware

Ask for a 1m context on an 8 GB card and ozgent works out what actually fits,
uses it, and tells you — instead of failing with a null pointer.

Three inference modes: `gpu` (fastest, smallest context), `gpu_ram` (bigger
context, slower), `ram` (no GPU at all).

```bash
ozgent web --inference-mode gpu_ram
```

→ [docs/settings.md](docs/settings.md)

---

## Docs

| | |
|---|---|
| [settings.md](docs/settings.md) | every option, and where to set it |
| [tools.md](docs/tools.md) | the built-in tools and the permission rules |
| [agents.md](docs/agents.md) | `@agents`: the built-in ones and making your own |
| [mcp.md](docs/mcp.md) | connecting MCP servers |
| [channels.md](docs/channels.md) | Telegram and WhatsApp |
| [api.md](docs/api.md) | the HTTP API, endpoint by endpoint |
| [install.md](docs/install.md) | deploying to another machine |

---

## Status

Early. It works, it is tested, and the interfaces still move.

`ozgent web` binds `0.0.0.0` by default so a phone on the same wifi can reach
it, and **has no password** — use `--host 127.0.0.1` on a network you don't
trust.

MIT licensed. `vendor/` carries patched copies of llama.cpp and its Rust
bindings, under their own licences — see [vendor/README.md](vendor/README.md).
