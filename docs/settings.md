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
| `--ctx`, `-c`, `--context` | `ctx` | `context_length` | context length in tokens |
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

Loading settings — `--gpu-layers`, `--no-gpu`, `--cpu-moe`, `--cache-type`,
`--control-vector` — are in `ozgent --help`. They differ in one way that
matters below.

## Why `/config ctx` says "applies when the model is next loaded"

Context length, layer placement, cache types and expert offload are fixed when
the weights are loaded, and cannot change under a live KV cache. `/config ctx`
therefore saves the value and tells you the one currently in force, rather than
printing a change that did not happen. It takes effect the next time that model
is loaded — a new `ozgent chat`, or `/models` switching away and back.

Everything else in the table above is per-turn and applies immediately, from
the next message on.

## Seeing what is in force

```
ozgent config show <model>    # resolved, all layers merged
ozgent show <model>           # manifest and settings together
/config                       # inside a chat, for the model you are in
/stats                        # context used, layers, what was reused
```
