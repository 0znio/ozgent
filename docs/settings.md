# Settings

Every generation setting exists in three places, with the same name and the
same meaning:

| where | example | scope |
|---|---|---|
| a command-line flag | `ozgent chat coder --ctx 32k` | this run only |
| `/config` inside a chat | `/config ctx 32k` | saved for this model |
| `config.toml` | `context_length = "32k"` | saved, editable by hand |

Later rows are the defaults; earlier ones override them. A flag never has to be
undone afterwards, which is the point of having it.

## Sizes

Anywhere a number of tokens is asked for, `k` and `m` work and mean the binary
multipliers — `8k` is 8192, not 8000, because 8192 is the number you meant.
`8K`, `8kb` and `8_192` all parse. So do `128k` and `1m`.

This holds on the command line and in `config.toml` alike:

```toml
[models."Qwen3.5-4B:Q4_K_M"]
context_length = "32k"     # or 32768; identical
max_tokens     = "2k"
```

A value that is not a whole number of tokens is refused rather than rounded:
`1.5k` is 1536, but `1.3k` is an error, since silently picking 1331 or 1332 is
worse than saying so.

## The settings

| flag | `/config` key | `config.toml` | what it does |
|---|---|---|---|
| `--ctx`, `-c`, `--context` | `ctx` | `context_length` | context length in tokens; 32k unless set, then lowered to what the model was trained on and what fits in memory |
| `--temperature`, `-t`, `--temp` | `temperature` | `temperature` | lower is more focused |
| `--top-p` | `top_p` | `top_p` | nucleus sampling |
| `--top-k` | `top_k` | `top_k` | consider only the K likeliest; 0 disables |
| `--min-p` | `min_p` | `min_p` | floor, relative to the likeliest token |
| `--repeat-penalty` | `repeat_penalty` | `repeat_penalty` | penalise tokens already used |
| `--max-tokens`, `-n` | `max_tokens` | `max_tokens` | cap on output; 0 means until it stops |
| `--seed` | `seed` | `seed` | with `--temp 0`, reproduces an answer |
| `--think` / `--no-think` | `thinking` | `thinking` | `auto`, `on`, `off` |
| `--effort` | `effort` | `reasoning_effort` | `low`, `medium`, `high` |
| `--no-tools` | `tools` | `tools` | whether the model may call tools |
| `--inference-mode` | `mode` | `inference_mode` | `gpu`, `gpu_ram`, or `ram` |
| `--no-kv-offload` | `kv` | `kv_offload` | keep the KV cache in RAM, not VRAM |

Loading settings — `--gpu-layers`, `--no-gpu`, `--cpu-moe`, `--cache-type`,
`--control-vector` — are in `ozgent --help`. They differ in one way that
matters below.

## The default model

`ozgent chat` with no model named, and the web interface on every visit, both
start with whichever model you have made the default:

```
ozgent default coder        set it
ozgent default              what it is now
ozgent default --clear      go back to naming one each time
/default                    inside a chat, pin the model you are in
```

It is also Settings → General in the web interface, and `default_model` in
`config.toml`. All four are the same value.

## The web interface and the admin page

```toml
[web]
admin_password_hash = "$argon2id$v=19$m=19456,t=2,p=1$…"   # written by `ozgent admin setup`
idle_unload_minutes = 15   # drop the model after this long with no questions; 0 never does

[tools]
handoff = true    # the model may pass a request to an @agent by itself
```

`idle_unload_minutes` matters most on a machine running
[the daemon](daemon.md), which is awake for twenty-three hours a day doing
nothing: a model held for one 9:20 brief holds several gigabytes until
midnight. The cost of dropping it is one reload — the same wait the first
question of the day pays anyway. Set `0` if you have VRAM to spare and use the
same model all day.

The admin password is never stored — only an Argon2id hash of it, set with
`ozgent admin setup` and replaced with `ozgent admin reset` if it is
forgotten. A password typed straight into this file is refused, not trusted.
`config.toml` is written readable only by you, since it can hold a bot token.

## When the context you asked for is not the one you get

A model's advertised window is a claim about which positions it understands,
not a promise that the cache for them fits in memory — at the million-token
windows recent models advertise, that cache runs to hundreds of gigabytes.
ozgent sizes the window to the memory that actually exists and says so:

```
Qwythos-9B-MTP:Q4_K_M · 50k ctx
· asked for 1m; that much KV cache does not fit in this machine's memory
```

`/config` shows both numbers when they differ, and the status line always shows
the one in force.

### Where the model runs

Two settings decide this — how many layers sit on the GPU, and whether the KV
cache sits with them — and the useful combinations are few enough to name:

```
ozgent chat coder --inference-mode gpu_ram
ozgent web --inference-mode gpu_ram
/config mode gpu_ram              # inside a chat; saved, applies on next load
```

| mode | weights | KV cache | window bounded by |
|---|---|---|---|
| `gpu` (default) | GPU | GPU | free VRAM |
| `gpu_ram` | GPU | system RAM | system RAM |
| `ram` | CPU | system RAM | system RAM |

`gpu` is the fast one, and the reason a model advertising 128k opens at 50k:
the window is whatever VRAM is left after the weights. `gpu_ram` keeps the
compute on the card and moves the cache off it, so the full window is usually
available — attention then reads the whole cache across PCIe on every token.
`ram` uses no GPU at all.

Measured on one machine, 200 tokens at a 32k window with a 9B model, so the
window is identical and only the cache location differs:

| mode | wall clock |
|---|---|
| `gpu` | 3.1s |
| `gpu_ram` | 4.9s |
| `ram` | 82.5s |

The gap between the first two widens as the window grows, because the cache
being read each token grows with it. On the same machine `--ctx 100k` gives
51200 tokens under `gpu` and the full 102400 under `gpu_ram`.

A mode is a shorthand, not an override: `--gpu-layers` or `--no-kv-offload`
set explicitly still win, so a placement you have tuned for your card is not
lost by naming a mode. ozgent suggests `gpu_ram` only when it would actually
help — on a machine whose RAM is no larger than its spare VRAM, moving the
cache buys nothing.

Cheaper to try first: quantise the cache (`--cache-type q4_0`), or close
whatever else is using the card.

A window cut for a different reason says so instead:

```
· asked for 256k; this model was trained for 128k
```

Asking for more than the training length is not a longer memory — it is a model
reading positions it has never seen — so that one is clamped too.

## Why `/config ctx` says "applies when the model is next loaded"

Context length, layer placement, cache types and expert offload are fixed when
the weights are loaded, and cannot change under a live KV cache. `/config ctx`
therefore saves the value and tells you the one currently in force, rather than
printing a change that did not happen. It takes effect the next time that model
is loaded — a new `ozgent chat`, or `/models` switching away and back.

Everything else in the table above is per-turn and applies immediately, from
the next message on.

## Messaging channels

`[channels]` is where Telegram and WhatsApp are configured. It is off, and
admits nobody, until deliberately changed — see [channels.md](channels.md),
which explains why that default is not merely cautious.

```toml
[channels]
enabled = true
model   = "coder"        # falls back to default_model

[channels.telegram]
enabled = true
token   = "…"            # or $OZGENT_TELEGRAM_TOKEN
allow   = ["@ada"]       # empty admits nobody
```

## Scheduled jobs

Jobs are **not** in `config.toml`. They live in ozgent's database, because they
are data rather than configuration and three surfaces edit them concurrently —
`ozgent scheduler`, the `/scheduler` page, and the model itself when you ask
for one in a chat. See [the scheduler](scheduler.md).

## MCP servers

`[mcp]` connects ozgent to Model Context Protocol servers, whose tools then sit
alongside its own. Off by default, and every tool from a server asks before it
runs unless that server is marked trusted — see [mcp.md](mcp.md), which
explains why a server's own `readOnlyHint` is not taken at face value.

```toml
[mcp]
enabled = true

[mcp.servers.files]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]
```

## Downloads

`ozgent pull` fetches a model as several ranges at once, which is several times
faster than one connection — Hugging Face serves a single stream at a few MB/s
however fast your link is.

| | |
|---|---|
| `$OZGENT_DOWNLOAD_CONNECTIONS` | how many at once (default 8, max 32) |

Set it to `1` for one connection, which is what a proxy or a rate-limited
mirror may want.

An interrupted download resumes. The file is fetched in 16 MB slices and a
`.part.ranges` file beside it records which ones landed, so stopping costs at
most the slices in flight rather than the whole file. A server that does not
serve ranges is detected and falls back to a single connection.

## Seeing what is in force

```
ozgent config show <model>    # resolved, all layers merged
ozgent show <model>           # manifest and settings together
/config                       # inside a chat, for the model you are in
/stats                        # context used, layers, what was reused
/permissions                  # what tools may do without asking
```

In a terminal the status line at the bottom carries the same facts as they
change: the model, how full the context is, the rate of the last reply,
thinking, tools, and the sampler. The bar above it says what tools are allowed
to do — and is where a tool asks, when one has to.
