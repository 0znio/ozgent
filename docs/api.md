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
network; callers from elsewhere then need a key.

## Authentication

Every request, on either port, is checked in this order. A refusal says which
step refused it.

1. **The address.** `[web.access]` `mode` is `open` (everyone except `deny`)
   or `allowlist` (only `allow`, plus this machine). Rules are IPv4 or IPv6
   addresses or CIDR ranges; an IPv4 client seen as `::ffff:a.b.c.d` is
   matched as IPv4. A refused address is dropped at `accept`. Behind a reverse
   proxy, list it in `trusted_proxies` and the right-most address in
   `X-Forwarded-For` that is not itself a trusted proxy is the client; nobody
   else's forwarding header is believed. → `403`
2. **The rate.** `requests_per_minute` per address (this machine is exempt),
   and `max_auth_failures` bad keys lock an address out for
   `lockout_minutes`. → `429` with `Retry-After`
3. **The name.** `Host` must be `localhost`, a bare IP address, or a name in
   `hosts`. This is what stops a web page reaching the server through a domain
   its owner points at your machine. → `421`
4. **The origin.** A browser request that changes something must come from
   this server's own pages. → `403`
5. **The caller.**

| caller | how | may use |
|---|---|---|
| a program on this machine | nothing, while `local_api_open` is on (the default) | `/v1`, as the owner |
| this machine's own page | the local token, sent as `x-ozgent-token` (the page is given it; the terminal reads `~/ozgent/run/local-token`, mode 600) | everything but `/api/admin/*` |
| another machine's browser | the admin session cookie, with `x-ozgent-admin: 1` on writes | everything |
| a program with a key | `Authorization: Bearer ozk_…` or `x-api-key: ozk_…` | `/v1`, within the key's scopes |

Keys are made on the admin page (or `POST /api/admin/keys`), shown once, and
stored as a SHA-256. Their scopes:

| scope | allows |
|---|---|
| `inference` | the models, with the caller's own tools; embeddings |
| `tools` | ozgent's own tools whose effect is *read* |
| `tools_write` | also tools that change files or data |
| `tools_execute` | also tools that run programs — still in the sandbox, and only if `[tools.config.permissions] shell` allows |
| `agents` | `@agents` |

A key's scopes stand in for the question the tool policy would ask; they never
lift a `deny`. A request outside them is `403` with `permission_error`.
`ozgent serve --api-key SECRET` still works: that one key has every scope.

A missing or wrong key is `401`.

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

`input` is a string or an array of up to 2,048 strings. Returns one
unit-length vector per input in OpenAI's shape, and `model` names the model
that made them.

| field | |
|---|---|
| `model` | an installed embedding model, or anything else — an OpenAI name like `text-embedding-3-small` gets the configured embedding model. Naming a *chat* model is a `400`. |
| `encoding_format` | `float` (default) or `base64` — little-endian f32, as OpenAI's SDKs expect |
| `dimensions` | keep the first N dimensions and renormalise, for Matryoshka-trained models such as Qwen3-Embedding |

Each input is embedded whole up to the model's trained window (32,768 tokens
for Qwen3-Embedding), or `[embedding] max_tokens` if that is lower. A longer
input is a `400` naming the limit, as OpenAI does, never a vector of its first
half. On the GPU a long input also has to fit in the memory free at the time.

Token arrays, and empty strings, are a `400`. With no embedding model
installed, or `[embedding] enabled = false`, the answer is a `400` saying so.

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

They belong to whoever owns the machine, so an API key does not open them.
The page carries the owner's local token (see [Authentication](#authentication));
from another device, sign in at `/admin` first and the session cookie
covers them too. Anything else gets `401`. The `/api/admin/*` and `/api/hub/*`
routes, and deleting a model, always need the admin session. `ozgent web`
binds `0.0.0.0` by default so a phone on the same network can reach it;
`--host 127.0.0.1` keeps it to this machine.

All bodies are JSON. Failures are `{"error": "..."}` with `400` when the
request was wrong, `404` when the thing asked for is not there, and `500` when
ozgent was wrong. The distinction is load-bearing rather than cosmetic: a
client that retries on `500` would keep resending a request that can never
succeed, and a log full of `500`s hides the ones that are actually ozgent's
fault.

## Pages and assets

| method | path | |
|---|---|---|
| `GET` | `/`, `/new`, `/chat` | the interface (same HTML; the browser routes) |
| `GET` | `/app.css`, `/app.js` | its assets, compiled into the binary |
| `GET` | `/admin`, `/admin.js` | the admin page |
| `GET` | `/scheduler`, `/scheduler.js` | scheduled jobs |
| `GET` | `/media/{name}` | an image that was attached to a message |

## Conversations, across all of them

### `GET /api/search?q=…&limit=30`

Full-text search over every message in every conversation. Ranked by bm25;
each hit carries enough to render a result and open the thread it is in.

```json
[{ "conversation": 4, "uuid": "…", "title": "the deploy script", "seq": 6,
   "role": "user", "snippet": "…the deploy script keeps timing out…",
   "created_at": 1755648000 }]
```

A query that matches nothing is `[]`, not an error, and a query full of FTS5
operators is neutralised rather than rejected — the box is free text and people
paste anything into it.

### `POST /api/conversations/{id}/rewind`

```json
{ "seq": 4 }
```

Drops that message and everything after it, and answers
`{ "removed": 2, "message": "the text that was at seq 4" }`. What "regenerate
this reply" and "edit and resend" both are underneath — one operation, so the
two cannot disagree. Facts extracted from the removed messages go too;
otherwise a fact learned from a turn that no longer exists keeps being recalled
as though you had said it.

### `GET /api/conversations/{id}/export`

The conversation as Markdown, with a `Content-Disposition` naming the file
after its title. Tool results are omitted — they are the machinery of a turn —
but each turn says which tools it used, so an answer full of current facts does
not read as invented.

## The scheduler

Every route here is open, like the rest of the interface: scheduling does
nothing the chat page does not already do. See [the scheduler](scheduler.md).

### `GET /api/scheduler`

Every job, plus what the page needs around them: whether this process is the
one running jobs (`hosted`), who is if not (`elsewhere`), which channels are
connected, this machine's timezone, and the agents and tools a job can be
given.

### `POST /api/scheduler`, `PUT /api/scheduler/{name}`

Create or change one. Every field is optional on a `PUT`, and only what is sent
is written — two surfaces edit these rows, and sending a whole job back would
let a stale page revert a change made from a chat a moment earlier. An empty
`agent` or `only_if` **clears** it; an absent one leaves it alone.

`400` for anything about the request that cannot work — an unreadable time, a
name already taken, a channel with nowhere to send to. `404` for no such job.

### `POST /api/scheduler/preview`

```json
{ "when": "every weekday at 9:20", "zone": "Asia/Kolkata" }
```

Reads a rule back and says when it would actually fire:

```json
{ "when": "cron 20 9 * * 1,2,3,4,5", "words": "every weekday at 09:20 UTC+5:30",
  "zone": "Asia/Kolkata",
  "fires": [{ "at": 1757648400, "in": "in 14 hours" }, …] }
```

The form calls this as you type. `20 9 * * 1-5` and `9 20 * * 1-5` are both
valid and only one of them is nine twenty in the morning.

### `GET /api/scheduler/{name}`

The job and its last thirty runs, each with its status, whether it was
delivered, and what it answered.

### `POST /api/scheduler/{name}/run`

Marks it due. Answers `{ "queued": "<name>", "hosted": true }` — `hosted` is
false when nothing is running jobs, in which case it stays queued rather than
running. It is never run from the handler: the scheduler runs jobs one at a
time, and this would be the one path that ignores that.

### `DELETE /api/scheduler/{name}`

Removes it and its history.

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

`effective` includes `style` and `system_prompt` — the model's response style
and persona; see *Styles* below.

### `PUT /api/models/{model}/options`

Body is an `Options` object — the same keys as `[models."name:tag"]` in
`config.toml` — and it **replaces** the model's overrides. A value equal to
what the model would inherit is dropped rather than stored.

### `PATCH /api/models/{model}/options`

Change some keys and leave the rest: `{"style": "concise"}` sets one,
`{"system_prompt": null}` clears one. An unknown style is a `400`. This is
what `/style`, `/persona` and `/effort` use.

## Styles

A model's **style** shapes how it answers, and its **persona**
(`system_prompt`) is standing instructions; both go into the system prompt of
ozgent's own chats — web, terminal, channels — and never into a `/v1` request.

### `GET /api/styles`

```json
{ "styles": [
  { "name": "concise", "title": "Concise — short answers, no padding", "prompt": "Answer concisely…", "custom": false },
  { "name": "pirate", "title": "Pirate", "prompt": "Answer like a pirate…", "custom": true } ] }
```

Built in: `concise`, `detailed`, `to-the-point`, `adhd`, `beginner`, `expert`,
`casual`, `formal`, `tutor`.

### `PUT /api/styles/{name}` · `DELETE /api/styles/{name}`

`{"title": "optional label", "prompt": "the instruction"}` makes or changes a
custom style, kept as `[styles.<name>]` in `config.toml`. Names are lowercase
letters, digits and hyphens and cannot be a built-in's. Deleting one clears it
from every model using it.

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
  "tools_off": ["run_command", "ask_agent"],
  "images": ["data:image/png;base64,…"] }
```

`tools_off` leaves those tools out for the main model this turn — what the
composer's Web switch and Tools tray send. An agent called by name keeps its
own tools regardless.

Each SSE `data:` line is one event, tagged by `type`:

| type | fields | |
|---|---|---|
| `loading` | `model`, `progress` | loading the weights, `progress` 0 to 1 |
| `ready` | `model`, `context` | the model is loaded; generation starts |
| `thinking` | `text` | a chunk of reasoning |
| `answer` | `text` | a chunk of the reply |
| `tool_call_started` | `name` | the model has committed to a call and is still writing it |
| `permission` | `id`, `name`, `arguments`, `effect` | waiting for you — answer with `/api/permissions/decide` |
| `tool_call` | `id`, `name`, `arguments` | the call is about to run |
| `tool_result` | `id`, `name`, `ok`, `summary`, `ms`, `detail` | how it went |
| `agent_start` | `name`, `description`, `tools`, `missing` | an agent has taken over — named by you, or handed the request by the model; events until `agent_end` are its |
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

## Getting models

Behind **Admin → Models**, and so behind the admin session (see below). The
same machinery as `ozgent pull`: the listing, the quantisation choice, the
parallel resumable downloader.

### `GET /api/hub/search?q=`

GGUF repositories on Hugging Face matching `q`, most downloaded first:
`{"results": [{"id", "downloads", "likes", "vision", "created_at"}]}`.

### `GET /api/hub/repo?repo=owner/name`

What a repository offers: `quants` (each with `quant`, `bytes`, `shards`,
`fits`, `context_fits`, `kv_cache`, `recommended`, and the `name` it would
install as), `vision` and `projector_bytes`, `gated`, `runnable` (false when
it has no GGUF at all), `context_train`, and the `gpu` sizes are judged
against. `fits` compares with the GPU's *total* memory at 85%, since the
loaded model is unloaded to make room. `recommended` is Q4_K_M when it fits,
otherwise the largest that does.

`context_fits` is how many tokens of context fit on the GPU beside that size,
and `kv_cache` the cache type that gets there — worked out from the model's
own header, read by byte range from the smallest file (a few hundred KB)
before anything is downloaded, with the same sizing the engine uses at load.
`null` when there is no GPU, the weights alone do not fit, or the header could
not be read.

### `POST /api/hub/pull`

`{"repo": "owner/name", "quant": "Q4_K_M", "alias": "qwen"}` — `quant` and
`alias` optional. Returns the new job immediately; the download runs on the
server and outlives the page. `409` if the same model is already downloading;
`400` if the alias is taken.

### `GET /api/hub/pulls`

Every job, newest first: `status` (`resolving`, `downloading`, `done`,
`failed`, `cancelled`), `model`, `done`/`total` bytes across all files,
`bytes_per_second` over the last few seconds, `file_index`/`file_count`,
`error`. Poll it; a reload or a second tab sees the same jobs.

### `DELETE /api/hub/pulls/{id}`

Cancels a running job — what arrived is kept, and pulling again resumes from
it — or clears a finished one from the list.

### `DELETE /api/models/{model}`

Deletes an installed model from disk, unloading it first if it is loaded.
Refused (`409`) while it is downloading. If it was the default model, the
default is cleared and the response says so (`"cleared_default": true`).

`GET /api/models` marks `embedding` models (read from the file's own header),
which the chat picker leaves out, and gives each model's `context_train`.

### `PUT /api/admin/models/upload?name=file.gguf` · `POST /api/admin/models/import`

Installing a GGUF from the browser's machine: the file is the raw request
body, streamed to disk and refused at the first four bytes if it is not GGUF.
Returns `{"id", "name", "bytes"}`. Then `{"weights": id, "mmproj": id|null,
"alias": "name"|null}` to `/import` installs it — a hard link on the same
filesystem, so no second copy — and returns `{"model", "vision"}`. Uploads
not installed within six hours are swept.

### `POST /api/admin/models/default`

`{"model": "name"}` — make it the default.

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

`choice` is one of `once`, `session`, `always`, `always_server` (every tool of the MCP server the tool comes from), `deny`, `deny_always`. `204`
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

`available` includes tools from MCP servers, named `<server>_<tool>`, each
with `"source": "mcp:<server>"`. They are read from the servers already
running, not started again for the page.

### `GET /api/tools/active`

The tools the running host has, cheaply — for the composer's tray:
`[{"name", "label"?, "description", "effect", "rule", "server"?}]`, where
`server` names the MCP server a tool comes from. Includes `ask_agent` when the
model may hand requests to agents.

### `GET /api/mcp`

The MCP servers and how they are doing, for the web page's tray and the
terminal's `/mcp`: `{"enabled", "tools_enabled", "reconnecting", "servers":
[{"name", "description", "enabled", "used", "sandboxed", "state", "error", "tools"}]}`,
where `used` is false for a server left out of the chats (`[tools] mcp_off`).
No settings: those are on the admin routes.

### `PUT /api/tools`

```json
{ "enabled": true, "search_provider": "brave", "api_key": "…",
  "disabled": ["run_command"], "mcp_off": ["github"] }
```

Every field optional. `disabled` leaves single tools out of every turn,
ozgent's own and MCP servers' alike; `mcp_off` leaves out every tool of the
named MCP servers, which keep running. A change to what the tool worker
reads (`enabled`, `disabled`, the search provider or key) restarts it, so it
takes effect on the next turn rather than the next restart; `mcp_off` needs
no restart. Keys are written to `config.toml` and never returned by `GET`.

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
to disk; the next turn picks up the new settings. `204` on success.

Secrets are masked. Any string under a key containing `key`, `token`,
`secret` or `password` reads as `••••••••`, and sending that mask back keeps
the stored value.

Some settings are never changed here, whatever `PUT` sends: `[channels]` and
`[web]` (the gateway, who may connect, API keys), `[mcp]`, the tool
interpreter and extra tool folders, and `[tools.config.permissions]`. Each
decides what program runs or who may reach this machine, so they change on
`/admin`, behind its password, or in the file.

---

# `/admin`

The gateway and model downloads, behind a password set on the machine with
`ozgent admin setup`. Stored as an Argon2id hash in `[web]
admin_password_hash`; the password itself is never stored.

Signing in sets an `ozgent_admin` cookie (`HttpOnly`, `SameSite=Strict`, 12
hours from the last use). Every request that changes something must also send
`x-ozgent-admin: 1`, which a form on another site cannot. Five wrong passwords
from one address lock it out for fifteen minutes, thirty from anywhere lock
everyone; `ozgent admin reset` on the machine sets a new password, signs every
browser out and lifts the lock.

| | |
|---|---|
| `GET /api/admin/session` | `{"configured", "signed_in", "invalid"}` — no session needed |
| `POST /api/admin/login` | `{"password"}` → sets the cookie; `401` wrong, `429` locked |
| `DELETE /api/admin/session` | sign out |
| `POST /api/admin/password` | `{"current", "new"}` — signs every other browser out |

### Gateway

| | |
|---|---|
| `GET /api/admin/gateway` | everything the page shows: each channel's settings and `runtime` (`phase`: `off`, `installing`, `starting`, `linking`, `connected`, `failed`, `elsewhere`; `who`; `detail`), the pairing code, the tools and their rules, the models |
| `PUT /api/admin/gateway` | `{"model": "name"|null}` — the model chats get |
| `PUT /api/admin/gateway/{telegram\|whatsapp}` | any of `enabled`, `allow` (the whole list; each entry checked), `tools` (`null` for all, a list for only those), `approve`, `stream`; WhatsApp also `self_chat`, `groups` |
| `POST /api/admin/gateway/telegram/token` | `{"token"}` — checked with Telegram, then saved; returns `{"bot"}` |
| `POST /api/admin/gateway/whatsapp/link` | installs the bridge if needed and starts linking |
| `GET /api/admin/gateway/qr` | the current linking code as SVG, while there is one |
| `POST /api/admin/gateway/{channel}/signout` | forget the token, or unlink the device on WhatsApp's side too |
| `POST /api/admin/gateway/{channel}/restart` | reconnect, clearing a failure |
| `POST /api/admin/gateway/pairing` | a new pairing code |

Each channel also takes `reply_unauthorized`: tell someone who is not on the
list that they are not, with the id you would need — at most once per person
every six hours, and to thirty people an hour at most. On by default for
Telegram, off for WhatsApp.

Changes apply to the running channels at once; there is nothing to restart.

### Security

| | |
|---|---|
| `GET /api/admin/access` | the `[web.access]` rules, the keys (id, name, scopes, created, disabled — never the key or its hash), the scopes there are, and `you`: the address this request came from |
| `PUT /api/admin/access` | any of `mode`, `allow`, `deny`, `trusted_proxies`, `hosts`, `local_api_open`, `requests_per_minute`, `max_connections_per_address`, `max_body_mb`, `max_auth_failures`, `lockout_minutes`. A rule that does not parse is a `400`; a change that would refuse your own address is a `409` unless `"force": true` |
| `POST /api/admin/keys` | `{"name", "scopes": [...]}` → `{"key", "id", ...}`. The key is in this response and nowhere else |
| `PATCH /api/admin/keys/{id}` | any of `name`, `scopes`, `disabled` |
| `DELETE /api/admin/keys/{id}` | revoke |
| `GET /api/admin/sandbox` | what tools may touch: `root`, `write`, `shell`, `shell_allow`, `shell_network`, `network`, `network_allow`, `network_private`, `allow_sensitive`; and, read-only, the interpreter and extra tool folders — each a program that runs as you, so changed only in `config.toml` — and the MCP servers' names (they have a section of their own below) |
| `PUT /api/admin/sandbox` | any of the editable ones; the tool host restarts to take them |

### MCP servers

Behind the admin password, because adding a server is running a program. The
environment and headers a server is configured with are returned by name only
(`"env": ["GITHUB_TOKEN"]`); a change sends only what changes. Every change
reconnects the servers in the background and returns at once: poll `GET` for
`reconnecting` and each server's `status`.

| | |
|---|---|
| `GET /api/admin/mcp` | `enabled`, `tools_enabled`, `reconnecting`, `runtimes` (`npx`, `uvx`, `docker`, `sandbox` available here), and `servers`: each with its settings (`command`, `args`, `url`, `env` and `headers` names, `sandbox`, `network`, `folders`, `trust_hints`, `timeout_seconds`, `only`, `source`, `description`, `load`, and `server_rule`: the rule set for all its tools, if any), its `status` (`state`: `connected`, `failed`, `disabled` or `off`; `error`; the last lines of its `log`; `server` name and version) and its `tools` (`name`, `remote`, `description`, `effect`, `offered`, the `rule` in force and its `own_rule`) |
| `PUT /api/admin/mcp` | `{"enabled": bool}`: use MCP servers at all |
| `POST /api/admin/mcp/reconnect` | reconnect every server |
| `POST /api/admin/mcp/servers` | add one by hand: `{"name", "kind": "command"\|"url"\|"npm"\|"pypi", "command"?, "args"?, "url"?, "package"?, "version"?, "env"?, "headers"?, "sandbox"? (default on), "network"?, "folders"?, "trust_hints"?, "preview"?}`. With `"preview": true` nothing is saved and the answer is `{"preview": "npx -y …", "sandbox"}`: the exact command |
| `PATCH /api/admin/mcp/servers/{name}` | any of `enabled`, `trust_hints`, `sandbox`, `network`, `folders`, `timeout_seconds`, `load` (`auto`, `always` or `on_request`; takes effect from the next message, without reconnecting), `args`, `only` (tools to offer by their name on the server; `null` for all), `env` and `headers` (a string sets, `null` removes, names not sent are kept), and `rules` (`{"<server>_<tool>": "allow"\|"ask"\|"deny"\|""}`, only for this server's tools; `"<server>_*"` sets the rule for all of them) |
| `DELETE /api/admin/mcp/servers/{name}` | remove it, its tools' rules and its sandbox home |
| `GET /api/admin/mcp/registry?search=&cursor=` | search the MCP Registry: each server's `name`, `title`, `description`, `version`, `repository`, a `suggested_name`, and its `options` (ways to run it), each with `kind`, `identifier`, `runner`, `supported`, `why_not`, `runner_present`, and the `inputs` it asks for (`key`, `name`, `required`, `secret`, `default`, `choices`) |
| `POST /api/admin/mcp/install` | `{"registry_name", "version"? (default latest), "option"?, "values": {"<input key>": "…"}, "name"?, "sandbox"?, "network"?, "folders"?, "preview"?}`. The entry is fetched again here and the command built from it; values for inputs it does not declare are refused |

### Models and memory

| | |
|---|---|
| `GET /api/admin/embedding` | `enabled`, `model` (`null` is automatic), `device` (`auto`, `gpu`, `cpu`), `max_tokens` (`0` is the model's own window), the installed embedding models, `status` (model, dimensions, where it runs, last error), `coverage` (messages with vectors, of all), `backfilling` |
| `PUT /api/admin/embedding` | any of `enabled`, `model`, `device`, `max_tokens`; the model reloads on its next use and earlier messages are re-embedded in the background |
| `POST /api/admin/embedding/backfill` | embed every message still missing a vector from the current model |
| `GET /api/admin/server` · `PUT /api/admin/server` | `idle_unload_minutes`, `parallel` (conversations per model at once, from the next load), `tool_timeout_seconds`, `max_calls_per_turn`, `handoff` |

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
