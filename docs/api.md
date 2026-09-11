# ozgent HTTP API

One API, spoken two ways. The same port answers **OpenAI** clients
(`/v1/chat/completions`) and **Anthropic** clients (`/v1/messages`), so
software written for either works unchanged: set its base URL to
`http://127.0.0.1:7337/v1` and pick a model from `/v1/models`.

| | command | default | what it is |
|---|---|---|---|
| **API** | `ozgent serve` | `http://127.0.0.1:7337/v1` | OpenAI- and Anthropic-compatible |
| **Web UI** | `ozgent web` | `http://127.0.0.1:7333` | the browser interface — and the same `/v1` API |

`ozgent web` serves `/v1` too, over the model it already has loaded. Running
`ozgent serve` alongside it would load a second copy into VRAM, so if the web
interface is up, point other programs at its port instead.

`ozgent serve` binds loopback only. `--host 0.0.0.0` exposes it to the
network, which you should pair with `--api-key`.

## Authentication

None by default. With `ozgent serve --api-key SECRET` (or `$OZGENT_API_KEY`),
every `/v1` request must carry it, in either client's spelling:

```
Authorization: Bearer SECRET
x-api-key: SECRET
```

A missing or wrong token is `401`.

## Errors

Each route answers in its own protocol's shape. On the OpenAI routes:

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
terminated by `data: [DONE]`. The final chunk carries `usage`; with
`"stream_options": {"include_usage": true}` a further chunk with empty
`choices` carries it too, as OpenAI sends it.

Several tool calls in one turn arrive with `index` 0, 1, 2…, which is what
OpenAI clients accumulate them by.

`"tool_choice": "none"` withholds every tool for that request, yours and
ozgent's.

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

Installed models, then agents as `@name`. Each entry carries both protocols'
fields — `object`/`created` for OpenAI, `type`/`display_name`/`created_at`
for Anthropic — and the list has Anthropic's `has_more`, `first_id` and
`last_id`, so either client reads it as its own.

## `GET /v1/agents`

Every agent in full — name, description, tools, rules — for a client that
wants to offer its own `@` menu.

---

# `/v1/messages` — Anthropic-compatible

## `POST /v1/messages`

Anthropic's Messages API. The official SDKs and anything written against
`api.anthropic.com` work with the base URL changed.

```bash
curl http://127.0.0.1:7337/v1/messages \
  -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model": "Qwen3.5-4B:Q4_K_M", "max_tokens": 1024,
       "messages": [{"role": "user", "content": "hi"}]}'
```

| field | notes |
|---|---|
| `model` | any name `ozgent list` shows, or `@agent` |
| `messages` | `user`/`assistant`; content a string or blocks |
| `system` | a string or text blocks |
| `max_tokens` | honoured; optional here (absent means the model's limit) |
| `temperature`, `top_p`, `top_k` | as usual |
| `stream` | Anthropic's event stream |
| `tools`, `tool_choice` | your tools, as on the OpenAI route |
| `thinking` | `{"type":"enabled","budget_tokens":N}`, `{"type":"disabled"}`, `{"type":"adaptive"}` |

Content blocks understood: `text`, `image` (base64; a URL source is refused
rather than ignored), `document` with a text source, `tool_use`,
`tool_result` (with `is_error`). Earlier `thinking` blocks are accepted and
not replayed. Anthropic's hosted tools (`web_search_20250305` and the like)
cannot run here and are not offered to the model.

**Thinking follows Anthropic's rule: absent means off.** A client that never
asks for thinking gets a model that answers directly. Asked for, reasoning
arrives as `thinking` blocks with an empty `signature`.

**Your tools.** When the model calls one, the reply ends with a `tool_use`
block and `stop_reason: "tool_use"`. Run it, send a `tool_result` block back
in the next `user` message, and continue.

**ozgent's tools.** `"ozgent_tools": true` and `native_tools` work exactly as
on the OpenAI route.

### Streaming

The standard sequence: `message_start`, then each block opened, filled and
closed by index (`content_block_start`, `content_block_delta` with
`text_delta` / `thinking_delta` / `input_json_delta`, `content_block_stop`),
then `message_delta` with `stop_reason` and `usage`, then `message_stop`.
`ping` keeps the connection alive while a model loads or a tool runs. Input
tokens are not known until the prompt has been read, so `message_start`
reports 0 and `message_delta` carries both counts.

### Errors

```json
{ "type": "error", "error": { "type": "invalid_request_error", "message": "…" } }
```

`invalid_request_error` (400), `authentication_error` (401),
`not_found_error` (404), `api_error` (500).

---

# Agents over the API

Agents work through both protocols without the client knowing they exist.

**By mention.** `@stock-guru` anywhere in the **latest** user message hands
that message to the agent. Only the latest: a mention further back in the
history your client resends has already been answered.

**By model.** Agents are listed in `/v1/models` as `@name`. Choose one as the
model and every message goes to it, running on the server's default model
(`ozgent default <model>`). Mentions in the message can add more agents after
it.

The agent runs server-side with **its own tools**, not yours, and under its
own rules: the built-in agents allow their tools, so they work over the API
where nobody can be asked. A tool whose rule is `ask` is refused — there is no
person on the other end of an API call — and the agent is told so.

What the agent did travels in the reasoning stream, which is where both
clients already show a model working:

- **OpenAI:** `reasoning_content`, as lines like
  `@stock-guru is working on this · tools: yahoo_finance, web_search`,
  `→ yahoo_finance action=quote symbol=NVDA`, `  ✓ 1 quote · 320 ms`,
  `@stock-guru finished · 3 tool calls · 12.4s`.
- **Anthropic:** the same lines in a `thinking` block, sent even when thinking
  was not asked for, because it is the account of what ran.

The agent's report is the answer: `content` on OpenAI, a `text` block on
Anthropic. A client that ignores reasoning still gets it.

For clients that want to draw their own agent view, OpenAI stream chunks for
agent and tool events carry an extra `ozgent` field with the structured event
(`agent_start`, `tool_call`, `tool_result`, `agent_end`), and a non-streamed
response lists them under `ozgent.events`. Clients that do not know the field
ignore it.

| to… | send |
|---|---|
| stop mentions calling agents | `"ozgent_agents": false` |
| structured output | `response_format` — mentions are not looked for, and choosing an agent as the model with it is a `400` |

## `GET /health`

No authentication. `{"status": "ok", "models": 4}` — for container and systemd
health checks.

---

# Web UI endpoints

Served by `ozgent web`. These are what the browser interface talks to. They are
**not OpenAI-compatible and not a stable API** — they change with the UI. Use
`/v1` for anything you want to keep working.

No authentication. `ozgent web` binds `0.0.0.0` by default so a phone on the
same network can reach it; `--host 127.0.0.1` keeps it to this machine.

All bodies are JSON. Failures are `{"error": "..."}` with `400` when the
request was wrong and `500` when ozgent was.

## Pages and assets

| method | path | |
|---|---|---|
| `GET` | `/`, `/new`, `/chat` | the interface (same HTML; the browser routes) |
| `GET` | `/app.css`, `/app.js` | its assets, compiled into the binary |
| `GET` | `/media/{name}` | an image that was attached to a message |

## Models

### `GET /api/models`

```json
[{ "reference": "Qwen3.5-4B:Q4_K_M", "alias": null, "quantization": "Q4_K_M",
   "size_bytes": 3413361504, "vision": true, "is_default": false }]
```

### `POST /api/unload`

Drops the loaded model and frees its VRAM. `204`, always.

### `GET /api/models/{model}/options`

Everything the engine exposes for one model, in three parts — which is the
point, because "what it runs with" and "what this model overrides" are
different questions:

```json
{ "model": "Qwen3.5-4B:Q4_K_M",
  "effective": { "temperature": 0.7, "context_length": 32768, "…": "…" },
  "overrides": { "context_length": 32768 },
  "limits":    { "context_length": 262144, "max_tokens": 262144 } }
```

`limits` is read from the GGUF header, so a slider cannot be dragged past what
the model was trained for.

### `PUT /api/models/{model}/options`

Body is an `Options` object — the same keys as `[models."name:tag"]` in
`config.toml`. Only the keys you send are changed; `null` clears one.

## Conversations

Conversations are shared with the terminal client and the messaging channels:
the same SQLite database, so a chat started anywhere shows up everywhere.

| method | path | |
|---|---|---|
| `GET` | `/api/conversations` | most recent first |
| `POST` | `/api/conversations` | `{"title"?, "model"?}` → the new conversation |
| `GET` | `/api/conversations/by-uuid/{uuid}` | look up by public id |
| `PATCH` | `/api/conversations/{id}` | `{"title": "…"}` |
| `DELETE` | `/api/conversations/{id}` | and its messages, facts and vectors |
| `GET` | `/api/conversations/{id}/messages` | the full history |

```json
{ "id": 3, "uuid": "8f14e45f-…", "title": "Arc versus Rc",
  "model": null, "created_at": 1788526764, "messages": 4 }
```

A message:

```json
{ "id": 7, "role": "assistant", "text": "…", "created_at": 1788526790,
  "thinking": "…",
  "tool_calls": [{ "name": "web_search", "arguments": {}, "ok": true,
                   "ms": 1180, "summary": "4 results", "detail": {} }],
  "media": ["a1b2c3.png"] }
```

`thinking`, `tool_calls` and `media` are omitted when empty. Use `uuid` in
links — row ids are sequential and leak how many conversations exist.

## Sending a turn

### `POST /api/chat` → `text/event-stream`

```json
{ "conversation": 3, "model": "Qwen3.5-4B:Q4_K_M", "message": "hello",
  "thinking": "on" | "off" | "auto",
  "tools": true,
  "images": ["data:image/png;base64,…"] }
```

Each SSE `data:` line is one event, tagged by `type`:

| type | fields | |
|---|---|---|
| `ready` | `model`, `context` | the model is loaded; generation starts |
| `thinking` | `text` | a chunk of reasoning |
| `answer` | `text` | a chunk of the reply |
| `tool_call_started` | `name` | the model has committed to a call and is still writing it |
| `permission` | `id`, `name`, `arguments`, `effect` | waiting for you — answer with `/api/permissions/decide` |
| `tool_call` | `id`, `name`, `arguments` | the call is about to run |
| `tool_result` | `id`, `name`, `ok`, `summary`, `ms`, `detail` | how it went |
| `agent_start` | `name`, `description`, `tools`, `missing` | an agent has taken over; events until `agent_end` are its |
| `agent_end` | `name`, `ok`, `ms`, `calls`, `rounds` | the agent finished; `ok` is false when it produced no report |
| `done` | `generated`, `tokens_per_second`, `prompt`, `prompt_ms`, `reused`, `stop` | |
| `error` | `message` | |

`tool_call_started` exists because a model writing a file spends the whole call
generating `content`. Without it the stream simply stops for a minute, which
reads as a dropped connection.

The reply is persisted by the server, not by the client: closing the connection
mid-generation loses the delivery, not the answer. Dropping it is also what
tells the engine to stop, so a closed tab frees the GPU.

A stored reply's `tool_calls` records each call and agent with `at`: where
in the reply text it happened, in UTF-16 units. The interface uses it to put
cards and agent blocks back between the right paragraphs on reload.

## Agents

### `GET /api/agents`

`{ "agents": [...], "errors": [...], "limits": {...} }`. Each agent has
`name`, `origin` (`builtin`, `user`, `override`), `description`,
`instructions`, `tools`, `permissions`, and optionally `max_rounds`,
`thinking`, `temperature`, `max_tokens`. `errors` lists agent files that could
not be read, and why.

### `PUT /api/agents/{name}`

Body: the agent without its name. Saving a built-in's name creates an
override. A definition that could not run — no instructions, a rule for a tool
it does not list, `max_rounds` out of range — is a `400` saying which.

### `DELETE /api/agents/{name}`

Deletes your agent, or your override of a built-in, which brings the original
back (`{"restored_builtin": true}`). A built-in itself cannot be deleted.

## Permissions

### `GET /api/permissions`

The rules joined against the tools actually installed — a tool with no rule of
its own still has an answer, the one its effect gives it:

```json
{ "defaults": { "read": "allow", "write": "ask", "execute": "ask", "unknown": "ask" },
  "tools": [{ "name": "write_file", "description": "Create or modify a file.",
              "effect": "write", "rule": "ask",
              "overridden": false, "granted": false }] }
```

`overridden` — set for this tool by name, rather than inherited from its
effect. `granted` — allowed for the rest of this run by a "don't ask again".

### `PUT /api/permissions`

```json
{ "read": "allow", "write": "ask", "execute": "deny", "unknown": "ask",
  "tools": { "run_command": "allow", "write_file": null },
  "clear_session": false }
```

Every field is optional. A tool mapped to `null` has its override **cleared**
and goes back to inheriting — a different state from being set to the value it
would have inherited. `clear_session` drops every "don't ask again this
session" answer.

### `POST /api/permissions/decide`

Answers a `permission` event. This is what unblocks the engine.

```json
{ "id": "early-write_file", "choice": "once" }
```

`choice` is one of `once`, `session`, `always`, `deny`, `deny_always`. `204`
when it landed, `404` when nothing was waiting for it — answered twice, or
after the wait ran out.

Unanswered questions are refused after five minutes, so a closed tab cannot
strand the engine.

## Tools

### `GET /api/tools`

```json
{ "enabled": true, "python": "python3", "timeout_seconds": 30,
  "max_calls_per_turn": 8,
  "available": [{ "name": "web_search", "description": "…", "enabled": true }],
  "search_providers": [{ "name": "brave", "configured": true, "needs_key": true }],
  "search_provider": "brave" }
```

`available` includes tools from MCP servers, named `<server>_<tool>`.

### `PUT /api/tools`

```json
{ "enabled": true, "search_provider": "brave", "api_key": "…",
  "disabled": ["run_command"] }
```

Every field optional. Changing any of them restarts the Python worker, so the
change takes effect on the next turn rather than the next restart. Keys are
written to `config.toml` and never returned by `GET`.

## Memory

Facts are what ozgent keeps about a conversation and retrieves into later ones.

| method | path | |
|---|---|---|
| `GET` | `/api/conversations/{id}/facts` | what is remembered |
| `POST` | `/api/conversations/{id}/facts` | `{"text", "scope"?, "pinned"?}` |
| `POST` | `/api/conversations/{id}/recall` | `{"query": "a question"}` → what memory *would* retrieve, without asking the model |
| `PATCH` | `/api/facts/{id}` | `{"pinned": true}` |
| `DELETE` | `/api/facts/{id}` | forget it |

```json
{ "id": 12, "text": "Prefers Rust over Go.", "scope": "conversation",
  "pinned": true, "created_at": 1788526764 }
```

`scope` is `conversation` or `global`. Pinned facts are always included,
whatever the question.

## Settings

### `GET /api/settings` · `PUT /api/settings`

The whole `config.toml`, as JSON.

`PUT` **replaces** it — send the document you got from `GET` with your edits,
not a fragment, or everything you left out goes back to its default. It saves
to disk and drops the loaded model, so the next turn picks up the new settings.
`204` on success.

Handle with care: this is the same document that holds your provider keys, your
channel allowlists and your MCP servers, and `GET` returns it in full.

---

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
