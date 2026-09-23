# How ozgent runs a model

This is the engineering side of ozgent: what happens between a message arriving
and a token leaving, and why each part works the way it does. It assumes you
know what a KV cache, a forward pass and a quantisation format are. Numbers
are measurements on the development machine unless stated otherwise: an RTX
5050 Laptop GPU (8 GB, sm_120, CUDA 13.3), a Ryzen 7 260 (8 cores / 16
threads, Zen 5), 22 GB of RAM.

Reference models used throughout:

| Model | Shape | Fits on 8 GB? |
|---|---|---|
| Qwen3.5-4B Q4_K_M | dense hybrid, 32 blocks (8 attention, 24 linear) | yes |
| Qwen3.6-35B-A3B IQ4_XS | MoE hybrid, 40 blocks + 1 NextN, 256 experts, 8 active | experts in RAM |
| Ternary Bonsai 27B Q2_g64 | dense hybrid, 64 blocks (16 attention, 48 linear), ternary weights | ~56 of 64 blocks |
| Spark-X2.5-4B Q4_K_M | dense, 36 blocks: 9 full attention, 27 on a 512-token sliding window; 256-wide heads | yes |

---

## 1. The stack

ozgent is a Rust program over llama.cpp, linked statically.

- **llama.cpp**: upstream release `v0.4.1`, vendored under
  `vendor/llama-cpp-sys-2/llama.cpp` with one local change (§12). The sys
  crate's build glue and the `llama-cpp-2` bindings are vendored too, and
  adapted to the newer C API. `vendor/llama-cpp-sys-2/OZGENT.md` records
  exactly what differs.
- **Engine** (`crates/ozgent-llama`): loading, placement, contexts, the
  decode hub, sampling, speculation, reasoning and tool-call parsing.
- **Daemon** (`crates/ozgent-web`): one inference thread per loaded model,
  serving the web UI, the terminal, the OpenAI/Anthropic-compatible API,
  messaging channels and the scheduler. The terminal is a client of it.
- **Tools** run in a separate Python worker over JSON-RPC on stdio, so a
  crashing tool cannot take the model down.

Everything below runs the same way whichever client asked.

---

## 2. Placement: deciding what goes on the GPU

`backend::Plan::for_model_with(path, opts, sequences)` decides, before
anything is loaded, how many transformer blocks go on the card and how many
blocks' experts go to system RAM. It reads only the GGUF header and tensor
table (`layout::read`); no weights are mapped.

### What a block costs

A block's cost on the card is its weights, **plus the recurrent state every
sequence keeps beside it** on a hybrid model. Qwen3.5-style linear-attention
layers keep a fixed-size state per conversation. It does not grow with
context, so no per-token figure ever showed it, but it is not small:

```
conv  = (ssm.conv_kernel - 1) × (ssm.inner_size + 2 × ssm.group_count × ssm.state_size)
state = ssm.state_size × ssm.inner_size          (floats, f32)
```

These are llama.cpp's own `n_embd_r` and `n_embd_s`. For Bonsai that is
817,152 floats per linear layer per sequence: 748 MB for five sequences. The
formula was checked against a failed allocation, where 47 GPU-resident linear
layers × 5 sequences × 817,152 × 4 bytes equals exactly the 768,122,880 bytes
llama.cpp tried to allocate. Before this was priced, Bonsai was placed 63 of
64 blocks deep and then no context of any size would open. The state lives
beside its block, so a block sent to the CPU takes its state with it; that is
why it is priced per block, times sequences.

### What must stay free

The plan reserves, before any weight is placed:

- **A window worth having**: cache at q8_0 for the largest of 64k, 48k, 32k,
  16k and the 8,192-token floor (`PLANNING_WINDOW`) that costs less than a
  tenth of the weights on the card. The window is elastic and the weights are
  not — reserving a full 107k window outright would leave room for four of a
  4B's 33 blocks — but leaving the window to whatever the weights happen to
  spare is just as wrong on a model near or above the size of the card: a
  session that cannot hold a document, several tool results and an agent's
  transcript fails the task however fast it decodes. Measured on
  Qwen3.6-35B-A3B, whose cache sits on one layer in four: planning for 64k
  moved one expert block of 41 to the host and opened the whole 64k window,
  with no measurable change in decode speed (medians of five alternating runs,
  18.9 against 19.5 tok/s, inside this laptop's own drift). A dense model
  whose cache costs gigabytes a window fails the tenth test and keeps its
  weights.
- **Compute scratch**, sized for the micro-batch the context will actually
  choose (§4), from a model learned per install.
- **The decode reserve**: memory llama.cpp allocates lazily on the first
  decode (cuBLAS workspaces, pool growth). It is measured on real loads and
  kept in `~/ozgent/cache/vram.json`, so estimates improve with use. Left out,
  a 23B MoE was placed with 15 MiB spare and its 32k window collapsed to 578
  tokens.
- **Fixed GPU tensors**: the output head and its norm.

Compute scratch is also what one setting buys back. llama.cpp sizes a
context's scratch for the worst case its batch allows — every row of a
micro-batch producing a full vocabulary of logits — and nothing here ever
asks for that: a prefill chunk wants one row and a verification wants the
confirmed token plus its draft. Capping it (`n_outputs_max`, a row per
conversation per drafted token, floor 16) took the scratch of a 4B at a
65,536-token window from 1,138 MiB to 632 MiB. Half a gigabyte, for nothing
given up — it becomes window, or the room a vision projector needs, or
another block of experts on the card.

A model's **vision projector** is not reserved for, because it is loaded only
when an image arrives — long after the window was sized. It is placed by what
is free at that moment: on the card when it fits, on the CPU when it does
not, which costs a few hundred milliseconds an image instead of the tens of
milliseconds it costs on the card. llama.cpp aborts the process rather than
failing when a buffer will not fit, and that is what the first image on a 4B
with a 107,008-token window did. Shortening the window or quantising the
cache harder to make room would cost every turn, image or not, and a model
whose weights already fill the card has nothing to give either way.

### Experts before blocks

For a mixture-of-experts model, routed experts leave the card before any
block does, and they leave by tensor rather than by whole layer: down, then
gate, then up. `_K` quantisations store `ffn_down_exps` at higher precision,
so "a third of a layer" is wrong by up to 20%. Moving one layer's experts
costs a MoE model little, because only 8 of 256 experts are read per token;
moving a whole block costs a full CPU matmul per token. Once anything is
evicted, the plan is made again with the heaviest block's experts held back,
because prefill uploads one block's evicted experts at a time into a buffer
llama.cpp sizes for the largest.

### The llama.cpp layer count

llama.cpp counts the output head as one more layer and fills from the top:
the first GPU block is `n_layer_all + 1 − n_gpu_layers`. ozgent used to pass
its block count straight through, which left block 0 on the CPU on every
model that fitted: a CPU matmul and a PCIe round trip on every token. That
was the entire gap to llama-server: 20.1 ms a token against 17.0 on
Qwen3.5-4B. A full plan now asks for everything; a partial plan adds one for
the head (`llama_gpu_layers`).

### Slots and the one-conversation rule

The daemon opens `web.parallel` conversation slots (default 4) plus one
sequence for the shared prefix (§6). On a hybrid model each costs its
recurrent state on every GPU block. For a model that does not fit, that
concurrency is paid for in blocks on the CPU: Bonsai with five sequences kept
50 of 64 blocks on the card and decoded at 5.4 tok/s. So when a plan with two
sequences keeps more blocks on the GPU than a plan with five, the model
serves one conversation at a time; others queue. Bonsai: 56 of 64 blocks,
12.4 tok/s. Models that fit, and MoE models whose blocks all fit, keep every
slot.

### When the plan is wrong anyway

The context is opened after the weights load, against real free memory
(§3). If the window that fits is below a floor (`min(context, trained
context, 8192)`), the engine reports `WindowBelowFloor` with how many bytes
were missing, instead of handing back a useless window. The daemon then
reloads, once:

- a MoE moves the experts of enough more layers to RAM to cover the shortfall;
- a dense model on automatic placement keeps that many fewer blocks on the GPU.

If no context opens at all, the same shortfall is reported when it can be
sized, so the same reload runs. A load that still fails is sent to the page
as an error rather than only logged. It used to leave the UI at "loading 100%"
indefinitely.

---

## 3. The context: KV cache and flash attention

`Engine::open` opens one `llama_context` for all slots, with a unified KV
cache (`kv_unified`), so conversations share one pool instead of each
reserving a full window.

### Flash attention

Requested as llama.cpp's AUTO policy, which checks whether the fused op lands
on the same device as its layer. The engine cannot know the outcome in
advance, so it reads llama.cpp's refusal: a quantised **V** cache requires
flash attention (K has no such rule). When the context is refused for that
reason, only V falls back to f16 and the open is retried at the full window.
An older rule gave up both halves and doubled the cache of every model
without flash attention for nothing.

### KV cache type

`auto` walks a ladder of (K, V) pairs, best first, and takes the first that
fits the budget at the requested window:

| With flash attention | bits/element | Without |
|---|---|---|
| q8_0 / q8_0 | 17.0 | q8_0 / f16 (24.5) |
| q8_0 / q5_1 | 14.5 | q5_1 / f16 (22.0) |
| q8_0 / q4_0 | 13.0 | q4_0 / f16 (20.5) |
| q5_1 / q4_0 | 10.5 | |
| q4_0 / q4_0 | 9.0 | |

f16 is deliberately not the top rung. Generation measured 48.3 tok/s at q8_0
against 48.5 at f16, which is noise, and the gigabyte f16 costs is worth more
as offloaded layers or window. Keys are held at higher precision than values
because attention is more sensitive to key error. Past the point where the
average cache read exceeds a quarter of the weight read (around 11k tokens on
a typical dense model), an f16 key is also the slower choice, and is skipped.

The cache is priced only on layers that keep one. A hybrid declares a scalar
`head_count_kv` and puts its layer pattern in `full_attention_interval`; taken
at face value that priced all 33 of a 4B's blocks when only 9 cache anything:
2176 MiB reserved where 550 were needed.

### Sliding-window layers

Some models attend over the whole context on only a few layers and over a
short sliding window on the rest. Spark-X2.5-4B runs 27 of its 36 layers on a
512-token window (the GGUF's `<arch>.attention.sliding_window_pattern`, one
bool per layer). Those layers never look further back than the window, so
their cache only has to hold it.

llama.cpp's C API defaults to `swa_full = true`, which gives every
sliding-window layer a cache as long as the whole context (its own `common`
tools default the other way). ozgent opens every context with
`swa_full = false` and prices the cache the way llama.cpp will size it: the
growing cache on the full-attention layers only, plus a fixed window cache of
`pad256(min(n_ctx, n_swa × n_seq + n_ubatch))` cells on the others
(`accel::swa_cells`), in the planner and the engine alike. Spark's heads are
256 wide, twice the usual, so this matters: at 32k, cache plus scratch went
from 2,800 MiB to 1,106 MiB, and a 64k window that had to fall to a 4-bit
cache now fits at 8-bit in 1,910 MiB. With four conversations at once, decode
went from 19.8 to 29.3 tok/s (+48%), because interleaved sequences no longer
make attention walk a mostly masked full-length cache: one pass at 16k with
two sequences, 37.1 → 26.7 ms. A single conversation alone is flat (58.4
against 56.9 tok/s); flash attention already skipped masked tiles when one
sequence's cells were contiguous.

A window cache recycles cells that have slid out of the window, which makes
three things that were safe on a full cache unsafe, each silently:

- **Going back a turn.** A trim to an earlier position leaves the window
  holding cells from after it, or missing ones before it. Measured: logits off
  by 13.9 and a different top token, with no error. Each checkpoint (§6) now
  saves the window's own state beside it (`PARTIAL_ONLY`), and a rewind is
  trim plus restore (`Session::rewind_window`): difference 0.000.
- **Rejecting a draft.** Trimming a rejected draft is exact only if no other
  pass ran between the verification and the trim, since another
  conversation's pass may have recycled the cells. The hub marks a windowed
  verification as unsettled and holds the next gather until it settles (at
  most 2 s), with one micro-batch per decode (`n_batch == n_ubatch`).
- **The shared prefix.** It is lent by `seq_cp`, and shared cells never
  recycle, so it would slowly fill the window. After it is filled it is
  compacted (a partial save and restore of its own sequence) down to the live
  window.

`OZGENT_SWA_FULL=1` restores llama.cpp's default, for comparison;
`examples/swabench` measures both and, with `SWA_CHECK=1`, checks the rewind
against a fresh decode.

### Fitting the window

The window is fitted to what is left, then opened. If llama.cpp refuses (its
own estimate of scratch differs), the engine **bisects** between the largest
window known to work and the smallest known to fail, rather than stepping by
a ratio: it lands within a few percent in three or four attempts, and each
attempt is only an allocation. Before shrinking the window, a widened
micro-batch is given up first (§4), because scratch scales with the
micro-batch and hardly at all with the window. Shrinking a 32k window to 960
tokens left a 1024-wide micro-batch's 973 MiB scratch in place, and then no
prompt fitted at all (`NoKvCacheSlot`).

The compute buffer size llama.cpp reports is recorded against the shape that
produced it, which is what makes the next placement estimate better.

---

## 4. Prefill and the micro-batch

A prompt is fed in batches of `n_batch` tokens, and llama.cpp runs each batch
as physical micro-batches of `n_ubatch`. Each micro-batch is a pass over the
weights, so a wider one means fewer passes and faster prefill, at the cost of
more compute scratch.

`Engine::open` weighs two candidates, the default 512 and `WIDE_BATCH` =
1024, and takes the wide one only if it costs no window. Measured: GLM prefill
316 → 503 tok/s, Qwen3.5-4B 1662 → 1923 tok/s.

With experts in RAM this matters more, because every micro-batch of a prefill
uploads the evicted experts across PCIe again. Halving the micro-batch count
nearly halves that. When the wide batch misses by a few layers' worth, the
load moves that many more experts to RAM to afford it, bounded at a twentieth
of the model: on the 35B, prefill went 378 → 589 tok/s for 26.2 tok/s decode
against 26.4. Four more layers and a 2048 batch cost 12% of decode, so that
is not done.

---

## 5. The decode hub

Every slot decodes through one `Hub`, which owns the context. Continuous
batching:

- A slot asks for its next token by queueing work. Whoever finds no driver
  becomes the driver and runs passes until its own answer is ready. Anything
  arriving mid-pass rides the next one.
- **Gather window.** Slots do not arrive together: each samples and
  detokenises before asking again, so a driver that ran immediately batched
  almost nothing (a flat 1.05×). The driver holds the pass open until every
  *running* slot has asked, up to half a pass's measured duration (250 µs to
  20 ms). A slot waiting on a tool is parked and not counted.
- **Batches are sorted by sequence id.** llama.cpp splits a unified batch
  into micro-batches wherever sequence ids stop increasing; in arrival order
  that was one micro-batch per slot, a pass over the weights each, and 2.0×
  of the win disappeared.
- A lone caller pays nothing for the machinery: the gather wait is 0.00 ms
  per token in the instrumentation.

**A full cache.** The KV pool is shared. When a decode finds no free cell
(`NoKvCacheSlot`), the hub clears the caches of conversations that are not
generating at that moment, then as a last resort the spare copy of the shared
prefix, and retries. llama.cpp finds room for a whole batch before writing
any of it, so a refused batch leaves the cache untouched. A session notices
its cache was taken (an eviction counter per slot) and comes back from its
checkpoints (§6).

---

## 6. Keeping what was already read

A chat turn resends the whole conversation. Most of it is already in the
cache, and the work is making sure it stays usable.

### Prefix reuse and why hybrids are hard

For attention layers, reuse means keeping the common prefix and trimming the
rest (`seq_rm`). Linear-attention layers cannot be trimmed: their state is
updated in place and has no per-position history. So on a hybrid model, any
turn that diverges even slightly from the cached tokens loses the recurrent
state entirely.

### Checkpoints

After prefilling history, the session saves the sequence's state to **host
memory** at the boundary before the generation prompt: `state_seq_get`,
serialised rather than device-resident, because a device-resident state is a
list of views that llama.cpp aborts on restoring once the cache has moved on.
The next turn restores the longest saved boundary that is a whole prefix of
its prompt and prefills only what follows.

- At least 256 tokens to be worth saving; up to 8 kept, least recently used
  evicted.
- Budget: an eighth of `MemAvailable` after counting the experts in RAM as
  fully resident. ggml's own figure for host memory on Linux is total RAM,
  which is no budget at all. Checkpoints used to be forbidden outright
  whenever experts lived in RAM, and every turn on the 35B re-prefilled the
  whole conversation: 5–10 s before each reply. A follow-up now starts in
  about a second.

### The shared prefix ("commons")

The system prompt and tool schemas are the same ~2,800 tokens at the front
of every conversation. They are held once, in a sequence of their own past
the conversation slots, and a new conversation takes a copy (`seq_cp`): for
attention cells a change of ownership, not recomputation. It has to be a
whole sequence, because llama.cpp's recurrent `seq_cp` copies the state as it
is now; copying a prefix of a longer conversation would hand over the state
of the wrong position.

- **Prewarm.** It is filled at load, not discovered on the second turn. The
  prewarm renders the real head of a turn (system prompt, tools, the agent
  handoff tool and its note) for two different user messages and holds their
  common token prefix. A first turn went from 1.6 s to 0.1 s to first token.
- **Discovery** compares prompts from *different* slots only. Comparing the
  same slot's successive prompts once promoted an agent's whole 6,740-token
  transcript to "the prefix every conversation starts with", and
  re-prefilled it for 3.5 s.

### A stable prefix

Anything that changes early in the prompt invalidates everything cached after
it. The system prompt used to carry the time to the minute, so any message
sent in a new minute re-read the whole conversation, the shared prefix and
every checkpoint. Now:

- the system prompt states only the **date and the user's time zone** (the
  machine's, e.g. `Asia/Kolkata (UTC+5:30)`, DST-aware via the zoneinfo
  database);
- each user message is stamped with the **local time it was sent**, taken
  from when it was stored, so an old message renders identically on every
  later turn;
- instructions ozgent adds mid-turn ("answer now", "out of room") are
  appended to the last message rather than sent as a late system message.
  Most chat templates fold late system messages into the first, which
  rewrites the top of the prompt.

---

## 7. Context management inside a turn

A turn can run up to 8 tool rounds, and agents up to their own limit.
Nothing bounded the sum of tool results: a research agent reached 30,899 of a
32,768 window and lost its answer to a full cache.

- **Per result**: a tool payload is cut to a character budget derived from
  the window the session actually opened, dropping whole results from the
  end of a list where the shape allows, so the JSON stays valid.
- **Per round**: before each round after the first, the prompt is rendered
  and tokenised. Above 75% of the window, older tool results (before the last
  assistant message) are cut to their first 1,200 characters with a note
  saying the model has already read them, oldest first, until the prompt is
  under 50%. The newest results stay whole. If one round's results alone are
  too big, the longest remaining result is halved until the prompt fits in
  85%. Stopping at half means it does not run again on the next round, which
  keeps the prefix stable.
- **No room left**: tools are withdrawn and the model is told to answer from
  what it has. If it calls a tool anyway, it is asked once more with every
  single-token call marker banned at the sampler (a logit bias of −∞), so
  the only thing it can write is the answer.

This is observation masking, which recent work finds as effective as
LLM-written summaries at a fraction of the cost, done only when needed,
because masking every turn costs accuracy on stronger models.

---

## 8. Sampling, reasoning and tool calls

- **Chat templates** are the model's own Jinja, rendered by minijinja, never
  llama.cpp's built-ins. `enable_thinking` is always stated: Qwen3.5 reads an
  undefined value as "off". When a template raises on a late system message,
  those are folded into the first and rendering is retried; the built-in
  fallback describes no tools, and a handoff once silently lost them that
  way.
- **Sampler chain**: optional grammar, optional logit bias, repetition
  penalty, then greedy at temperature 0, or top-k → top-p → min-p →
  temperature → seeded draw. The candidate array (152k entries) is one buffer
  per session, refilled in place. Rebuilding it per token cost 2 ms of a
  21 ms pass.
- **Reasoning effort**: a budget of 2,048 (low) or 8,192 (medium) reasoning
  tokens, or unbounded (high). When a budget runs out, the closing tag is
  decoded *for* the model, so it reads its own reasoning as finished and
  answers, instead of being cut off inside an open block.
- **Tool calls** are parsed in the model's native syntax (`<tool_call>`
  JSON, `<function=…>` XML-ish, `[TOOL_CALLS]`, `<|python_tag|>`). For
  models without a native format, a grammar derived from the tools' JSON
  schemas is switched on the moment a call opens, so the body cannot be
  malformed, and off again after. A call's name is announced to the client as
  soon as it is readable, long before its arguments are complete.
- **Parallel tool calls** run concurrently. Each result is shown the moment
  its call finishes; the model reads them in its own order. `web_search`
  spaces requests per provider (about one a second for Brave and DuckDuckGo,
  `min_interval` to override) so a batch of searches does not trip a rate
  limit.

---

## 9. Speculative decoding

Speculation proposes several tokens and verifies them in one pass. At
temperature 0 it must produce the text plain decoding would. The one
exception is inherent to batching: where the model's top two choices are
within a few hundredths of a logit, a two-token batch's float order can pick
the other, exactly as a prefill's can.

- **n-gram drafting** (`ngram.rs`): longest-match lookup in the context, a
  short key ranked by how far the match extends backwards, adaptive draft
  length, and a rolling acceptance rate that stops drafting when it does not
  pay and probes again later.
- **MTP / NextN** (`mtp.rs`): the model's own trained draft head, the default
  (`--spec auto`) whenever the GGUF carries one (`<arch>.nextn_predict_layers`,
  e.g. `unsloth/Qwen3.5-4B-MTP-GGUF`). One drafted token per round. Three
  things make it pay where the first attempt lost:
  - the head's layers are loaded (`load_mtp`), and its context, which keeps a
    KV cache of its own, absorbs every position the model decodes, fed the
    model's hidden state at the position before;
  - a hybrid model's context gets llama.cpp's `n_rs_seq` ring, so a rejected
    draft is trimmed instead of snapshotted and replayed. The ring is only
    exact straight after a multi-token batch, so nothing but draft
    verification ever trims a hybrid model's cache;
  - the hub owns one drafter per context and gathers the draft steps of every
    conversation into one decode, since each step reads the whole output head.

  Measured on Qwen3.5-4B-MTP, same process, drafting off against on: code
  +42%, math +42%, Hindi +33%, prose +27%, text identical. Four concurrent
  conversations: +40-47% total. Qwen3.6-35B-A3B-MTP with experts in system
  RAM: code +36%, prose +18%. The tool-call gate stays live: when a call
  begins, its grammar goes in and drafting stops for the rest of the turn.
- **Hybrid models without a head** still cannot trim a rejected draft: the
  recurrent state has already absorbed it. There, n-gram drafting snapshots
  the sequence state on the device, which is only sound when one slot has the
  context to itself, and only attempted when free VRAM covers the snapshot:
  llama.cpp asserts rather than failing when it cannot allocate one, which
  killed the process on Bonsai's first token.
- **DSpark / DFlash drafters** (e.g. Bonsai's) are not supported: upstream
  llama.cpp cannot load the `dspark` architecture, and the installer never
  offers a drafter as a model.

---

## 10. CPU threads

llama.cpp defaults to 4 threads. No constant is right, and it cannot be found
by comparing runs: the same setting loaded twice on the 35B measured 38.9 and
48.4 ms a token, because power, thermal state and page cache move a whole run
by more than threads ever do.

`threads::Tuner`, per hub, when no count is configured:

1. **Warm-up**: 64 decode passes untimed. Straight after a load the weights
   are still coming off disk (58 ms against 38 warm).
2. **Round robin**: passes alternate between half, three quarters and all of
   the *free physical* cores. Free means physical cores (from
   `/sys/.../topology`) minus what other processes used since the last round
   (`/proc/stat`). Each switch skips one pass before timing, and each
   candidate is timed 16 times. Alternating pass by pass means a growing
   context slows every candidate alike.
3. **Choice**: the fastest mean wins by a 3% margin; ties go to fewer threads.
   Hardware threads beyond the physical cores are never tried: two threads on
   one core fight for its memory bandwidth, and 16 was slower than 8 on every
   model.
4. **Re-measure** every 4,096 passes, sized to the cores free then.

Only decode passes are timed; prefill gets every free physical core. It
changes nothing about output. Chosen here: Qwen3.5-4B 4 (no difference), the
35B 6–8, Bonsai 6.

---

## 11. Ternary Bonsai

Bonsai is Qwen3.6-27B with ternary weights ({−1, 0, +1} in 2-bit slots, an
f16 scale per group). The repository ships several packs, and they are not
interchangeable:

| Pack | GGUF type | Loads in upstream llama.cpp |
|---|---|---|
| `Q2_g64` | 42 (Q2_0), 64-value groups | yes, and this is the one to use |
| `Q2_0` | 42, but 128-value groups | no: same type id, different layout |
| `PQ2_0` | 142 | no: type only Prism's fork defines |
| `dspark-*` | a speculative drafter | never on its own (no tokenizer) |

`Q2_g64` runs correctly on upstream with no special transform: output is
coherent. The installer recognises these pack names and excludes drafters,
which is how a drafter was once installed as "the model" and failed with
`unknown model architecture: 'dspark'`.

On this card the weights (7.05 GiB) nearly fill the ~7.4 GiB free, so 56 of
64 blocks fit beside a usable context, with one conversation at a time (§2).
12.4 tok/s, against llama-bench's 9.0 at the same split. All 64 on the card
would be ~22 tok/s, but only with a context too small to use.

---

## 12. The CPU kernel for Q2_0

Upstream has CUDA kernels for Q2_0 but only a scalar loop on x86, so every
Bonsai block on the CPU ran at scalar speed. `ggml_vec_dot_q2_0_q8_0` now has
an AVX2 path (`ggml-cpu/arch/x86/quants.c`, the one local change to
llama.cpp):

- One 16-byte Q2_0 half-block holds 64 weights, four per byte. AVX2 cannot
  shift each byte by a different amount, so the 8 bytes are broadcast to four
  64-bit lanes and shifted by 0, 2, 4 and 6 (`_mm256_srlv_epi64`). Masking
  with 3 leaves plane *e* of byte *b* at position 8e + b: weight 4b + e.
- The 32 activations are permuted to match: one in-lane `pshufb` and one
  cross-lane `vpermd`.
- Codes minus one give {−1, 0, 1, 2}; `maddubs` via sign tricks sums the
  products; one fused multiply-add per 32 values applies both scales.

5.3× faster on a 5,120-wide row (207 vs 1,100 ns), equal to the scalar
result within float rounding (2.3e-4 relative, over 2,000 random rows).

## 13. Embeddings

Memory recall (`ozgent-memory`) fuses keyword search with vector similarity.
The daemon serves both `/v1/embeddings` and that recall from one embedding
model, loaded beside the chat model (`crates/ozgent-llama/src/embed.rs`).

- **As the model was trained.** Qwen3-Embedding pools the last token and
  expects it to be the end-of-sequence token (`add_eos_token`), and expects
  queries to carry an instruction line; documents carry none. Neither was
  done, and texts were tokenised without special tokens. On a test of eight
  questions over twenty notes with near-miss decoys, the right note came
  first 7 of 8 times (MRR 0.938); with both it is 8 of 8 (MRR 1.000), with a
  wider margin over the best wrong note. Recall asks as `Role::Query`, stored
  messages are `Role::Document`, and the API treats its input as documents.
- **The whole text.** Texts were cut at 512 tokens. The limit is now the
  model's own trained window (`embedding.max_tokens`, 0 = that), and the
  context is opened at a working 8,192 and grown only when a longer text
  arrives, then released. A long note whose answer was in its last sentence
  ranked 5th of 21 at 512 tokens and 1st with the whole window. Batches are
  packed by tokens, not by count.
- **No logits for a whole long text.** An embeddings context makes llama.cpp
  reserve an output row, a full vocabulary of floats, for every token in a
  micro-batch. At 20k tokens that ended with the kernel killing the process.
  With last-token pooling the text is decoded with embeddings off except for
  its final micro-batch, which is all the pooled vector reads: cosine
  1.000000 against a single-pass decode. Micro-batches are capped at 2,048 so
  attention scratch stays bounded.
- **Placement.** On automatic, the embedder goes on the GPU only when a chat
  model is resident and it fits beside it with an 8k cache and the decode
  reserve; otherwise on the CPU, where it costs about 115–220 ms a short
  message against 9–13 ms on the card. `embedding.device` pins it.
- **Off the reply's path.** A stored message is embedded in the background
  after it is saved, outside the database lock. A question is embedded only
  when the conversation is longer than the recent window or facts carry
  vectors, and before the lock is taken. Vectors from another model are not
  comparable at any width: the model in use is recorded beside the database,
  and a change clears the vectors and re-embeds the history in batches of 16.

---

## 14. Downloads

Model files download as up to several parallel ranges into a preallocated
`.part` file, with a ledger of finished slices, so an interrupted download
resumes. Each slice retries with backoff for about two minutes of network
loss, a connection silent for 45 s is abandoned and retried, and a slice
that closes short is retried rather than counted done. The file is verified
against its sha256 before being renamed into place.

---

## 15. Measured results

Through the daemon, the path the web UI, terminal and API all use:

| | Before | After | llama.cpp reference |
|---|---|---|---|
| Qwen3.5-4B decode | 46 tok/s | 57.9 tok/s | llama-server 58.5 |
| Qwen3.5-4B first token, new conversation | 1.6 s | 0.1 s | |
| Qwen3.6-35B decode | 22.9 tok/s | 27.9 tok/s | llama-bench 26.4 |
| Qwen3.6-35B follow-up, first token | 5–10 s | ~1–2 s | |
| Ternary Bonsai 27B | did not load | 12.4 tok/s | llama-bench 9.0 (same split) |
| Spark-X2.5-4B, four conversations | 19.8 tok/s | 29.3 tok/s | |
| Spark-X2.5-4B, cache + scratch at 32k | 2,800 MiB | 1,106 MiB | |

With the model's own draft head (§9), on the GGUFs that carry one:

| | Plain | Drafting | Drafts accepted |
|---|---|---|---|
| Qwen3.5-4B-MTP, code | 57.0 tok/s | 80.8 tok/s | 88% |
| Qwen3.5-4B-MTP, prose | 56.8 tok/s | 72.2 tok/s | 69% |
| Qwen3.5-4B-MTP, four conversations at once | 133 tok/s | 188 tok/s | |
| Qwen3.6-35B-A3B-MTP, code | 27.9 tok/s | 38.1 tok/s | 94% |

Greedy text is identical to plain decoding in every case above, over four
prompts and three-turn conversations.

How these were measured, and the traps: never compare separate runs on this
laptop; compare within a run. Never time the first turn after a load. A
single-slot benchmark says nothing about the multi-slot daemon.
`tok/s` over a multi-round turn includes tool time, so compare wall time
when token counts differ.
