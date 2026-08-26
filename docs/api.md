# ozgent HTTP API

Two servers, deliberately separate.

| | command | default | what it is |
|---|---|---|---|
| **API** | `ozgent serve` | `http://127.0.0.1:7337` | OpenAI-compatible, for clients and scripts |
| **Web UI** | `ozgent web` | `http://127.0.0.1:7333` | the browser interface and the endpoints behind it |

`ozgent serve` is the one to point other software at. Set the base URL to
`http://127.0.0.1:7337/v1` and use any model name `ozgent list` shows.

Both bind to loopback only. `--host 0.0.0.0` exposes them to the network,
which you should pair with `--api-key`.

## Authentication

None by default. With `ozgent serve --api-key SECRET` (or `$OZGENT_API_KEY`),
every `/v1` request must carry it:

```
Authorization: Bearer SECRET
```

A missing or wrong token is `401` with an `invalid_request_error`.

## Errors

Errors are OpenAI-shaped:

```json
{ "error": { "message": "…", "type": "invalid_request_error", "code": null } }
```

| status | type | when |
|---|---|---|
| `400` | `invalid_request_error` | malformed body, or a prompt longer than the context |
| `401` | `invalid_request_error` | missing or wrong bearer token |
| `404` | `not_found_error` | no such model |
| `422` | — | the body did not parse as JSON of the expected shape |
| `500` | `api_error` | anything else |

A prompt that exceeds the context window is a **400**, not a 500 — it is the
caller sending too much, and retrying it unchanged will never succeed:

```json
{ "error": { "message": "the prompt is 40016 tokens but the context holds 32768; raise --ctx or shorten it" } }
```

---

# `/v1` — OpenAI-compatible

## `POST /v1/chat/completions`

The main endpoint.

### Standard fields

| field | type | notes |
|---|---|---|
| `model` | string | **required.** Any name from `ozgent list`, e.g. `Qwen3.5-4B:Q4_K_M` |
| `messages` | array | **required.** `{role, content}`; roles `system`, `user`, `assistant`, `tool` |
| `stream` | bool | `false` by default. See [Streaming](#streaming) |
| `max_tokens` | int | cap on generated tokens. `max_completion_tokens` is accepted as a synonym |
| `temperature` | float | `0` is greedy and reproducible |
| `top_p`, `top_k`, `min_p` | | sampling |
| `seed` | int | with `temperature: 0`, the same seed reproduces the same answer |
| `presence_penalty`, `frequency_penalty` | float | mapped onto llama.cpp's repeat penalty |
| `stop` | string or array | stop sequences |
| `tools`, `tool_choice` | | your own tools, offered to the model and handed back to you |
| `response_format` | object | see [Structured output](#structured-output) |

**Every one of these is per-request.** Two clients using one model do not
share sampling settings.

### ozgent extensions

| field | type | what it does |
|---|---|---|
| `reasoning` | bool / string / object | `false` or `"off"` suppresses reasoning; `"on"`, `"auto"` |
| `reasoning_effort` | `"low"` `"medium"` `"high"` | how hard a reasoning model should think |
| `ozgent_tools` | bool | let the model use ozgent's own tools (file access, web search) |
| `native_tools` | array of strings | restrict `ozgent_tools` to these names |

### Reasoning

Reasoning models return their scratchpad separately, so it never contaminates
the answer:

```json
{ "choices": [ { "message": {
    "role": "assistant",
    "reasoning_content": "The user is asking …",
    "content": "42"
} } ] }
```

`reasoning_effort` is passed to the model's own chat template where the model
understands it, so the model decides how long to think rather than being cut
off mid-thought. `high` is unbounded. Models whose template has no such
control — Qwen3.5 among them — cannot vary it; a token budget then acts only
as a backstop against a reasoning loop.

To turn reasoning off entirely:

```json
{ "model": "…", "messages": [ … ], "reasoning": false }
```

### Structured output

```json
{
  "model": "Qwen3.5-4B:Q4_K_M",
  "messages": [ { "role": "user", "content": "Give me a person." } ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
        "required": ["name", "age"]
      }
    }
  }
}
```

The schema is compiled to a grammar and enforced during sampling, so `content`
parses as JSON matching it. It is a constraint, not a request — the model
cannot produce anything else.

`{"type": "json_object"}` is also accepted for valid-JSON-without-a-schema.

### Tools

Two independent mechanisms, usable together.

**Your tools.** Pass `tools` in OpenAI's format. When the model calls one, the
turn ends with `finish_reason: "tool_calls"` and you run it:

```json
{ "choices": [ { "finish_reason": "tool_calls", "message": {
    "role": "assistant",
    "tool_calls": [ { "id": "call_…", "type": "function",
        "function": { "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}" } } ]
} } ] }
```

Send the result back as a `tool` message and continue.

**ozgent's tools.** `"ozgent_tools": true` lets the model use the Python tools
installed alongside ozgent — reading files, web search — and ozgent runs them
itself, in-process, before answering. You get the finished answer.

Restrict them with `native_tools`:

```json
{ "ozgent_tools": true, "native_tools": ["web_search"] }
```

That offers only `web_search`; anything not named is withheld, so the model
cannot read files even if it wants to.

> Tool calling is verified against Qwen3.5. Models with a different native
> tool-call syntax may not produce usable calls.

### Streaming

`"stream": true` returns `text/event-stream` in OpenAI's chunk format,
terminated by `data: [DONE]`. The final chunk carries `usage`.

### Usage and timings

```json
{ "usage": {
    "prompt_tokens": 9417,
    "completion_tokens": 150,
    "total_tokens": 9567,
    "timings": {
      "prompt_tokens": 28,
      "prompt_ms": 112,
      "prompt_tokens_per_second": 250.0,
      "completion_tokens": 150,
      "tokens_per_second": 56.4,
      "completion_ms": 2659,
      "cached_prompt_tokens": 9417
    }
} }
```

`cached_prompt_tokens` is how much of the prompt was already resident and did
not need reading again — it is what makes a long conversation cheap after the
first turn. `prompt_tokens` inside `timings` counts only what was actually
processed, while the outer one counts the whole prompt.

## `POST /v1/completions`

Plain text completion, no chat template. `prompt` instead of `messages`;
otherwise the sampling fields are the same.

## `POST /v1/embeddings`

```json
{ "model": "Qwen3-Embedding-0.6B", "input": ["hello world", "goodbye world"] }
```

`input` is a string or an array of strings. Returns one vector per input in
OpenAI's shape. An empty input is a `400`.

## `GET /v1/models`, `GET /v1/models/{model}`

Installed models in OpenAI's listing shape.

## `GET /health`

No authentication. `{"status": "ok", "models": 4}` — for container and systemd
health checks.

---

# Web UI endpoints

Served by `ozgent web` and used by the browser interface. Not
OpenAI-compatible and **not a stable API** — they change with the UI.

| method | path | purpose |
|---|---|---|
| `GET` | `/`, `/new`, `/chat` | the interface |
| `GET` | `/app.css`, `/app.js` | its assets |
| `POST` | `/api/chat` | send a turn |
| `GET` | `/api/models` | installed models |
| `POST` | `/api/unload` | release a model from memory |
| `GET` `POST` | `/api/conversations` | list, create |
| `DELETE` `PATCH` | `/api/conversations/{id}` | delete, rename |
| `GET` | `/api/conversations/{id}/messages` | history |
| `GET` | `/api/conversations/by-uuid/{uuid}` | look up by public id |
| `GET` `POST` | `/api/conversations/{id}/facts` | remembered facts |
| `POST` | `/api/conversations/{id}/recall` | preview what memory would retrieve |
| `PATCH` `DELETE` | `/api/facts/{id}` | pin, forget |
| `GET` `PUT` | `/api/settings` | global settings |
| `GET` `PUT` | `/api/models/{model}/options` | per-model settings |
| `GET` `PUT` | `/api/tools` | tool configuration and provider keys |
| `GET` | `/media/{name}` | an uploaded image |

---

# Examples

```bash
# Simplest possible call
curl http://127.0.0.1:7337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B:Q4_K_M",
       "messages":[{"role":"user","content":"Say hello"}]}'

# Reproducible, no reasoning
curl http://127.0.0.1:7337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B:Q4_K_M",
       "messages":[{"role":"user","content":"What is 27*43?"}],
       "temperature":0,"seed":7,"reasoning":false}'

# Let the model search the web, but nothing else
curl http://127.0.0.1:7337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B:Q4_K_M",
       "messages":[{"role":"user","content":"What happened in Oslo today?"}],
       "ozgent_tools":true,"native_tools":["web_search"]}'
```

With the official Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7337/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen3.5-4B:Q4_K_M",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(reply.choices[0].message.content)
```

`api_key` must be set to something because the client insists on it; ozgent
ignores it unless `--api-key` was given.

---

# Diagnostics

Every entry point — `chat`, `web`, `serve`, one-shot commands — writes to one
log file, which is what makes it useful when the server has been running under
systemd and something went wrong hours ago.

```bash
ozgent logs --follow      # watch it
ozgent logs --lines 200   # the last 200 lines
ozgent logs --path        # where it is, for tail or journald
```

The file records at `info` and rolls at 8 MiB, keeping three older files. The
terminal stays quiet unless you pass `-v`. `OZGENT_LOG` overrides both, e.g.
`OZGENT_LOG=ozgent_llama=debug`.
