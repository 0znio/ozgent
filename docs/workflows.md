# Workflows

A workflow is steps wired together: ask the model, use a tool, branch on the
answer, do something with the result. Draw one on the canvas, run it by hand, on
a schedule, or when something calls its URL.

    ozgent web        →  Workflows

## What a step can be

| | |
|---|---|
| **When I press run** | you start it, from the Run button |
| **On a schedule** | every so often, or once a day |
| **When called** | something POSTs to a URL |
| **Ask the model** | a prompt; the reply is the output |
| **Use a tool** | any tool you have installed |
| **Only if** | sends the run down one of two paths |
| **Build some text** | stitch earlier outputs into a piece of text |

There is no separate connector system, and that is deliberate: ozgent already
has a plugin system with typed schemas — the Python tools. Every installed tool
is a step, and the settings panel builds its form from the schema the tool
already declares. Drop a tool in `~/ozgent/tools`, reload the page, and it is a
node with the right fields on it.

## Passing data between steps

Every step produces something, and later steps read it by the step's **id** —
shown in the panel next to its name, and unaffected by renaming the step:

```
{{ search.results[0].title }}
{{ start.city }}
```

A reference that is the whole field keeps its type, so `{{ start.count }}` in a
field expecting a number stays the number 3 rather than becoming the string
"3". A reference with text around it becomes text.

A reference that does not resolve is **left standing** rather than blanked. A
prompt containing `{{ serach.results }}` shows you the typo; a blank would look
like the step returned nothing.

This is not an expression language. There is no arithmetic, no function calls,
and no way to reach anything but the outputs of steps that have already run. A
canvas invites pasting in things you were sent, and an evaluator is the shortest
path from "paste a template" to "arbitrary code runs on the machine hosting it".

## Branching

**Only if** compares two things and sends the run out of its `yes` or its `no`
port. Steps on the path not taken are marked *not reached* rather than hidden,
so a run shows the path it took.

Comparison is loose about numbers: `3` and `"3"` are equal. A value from a
webhook is a string and the same value from a tool is a number, and a condition
that quietly went the wrong way depending on where it came from is an extremely
hard bug to see.

## Tools, and what a workflow may not do

A step's arguments are a template, and a webhook fills that template with data
from whoever called it. So `run_command` with `{{ trigger.cmd }}` in it is a
remote shell, written entirely on a canvas, with no prompt anywhere.

A workflow therefore runs under the same rule as the OpenAI-compatible API:
**there is nobody to ask, so anything that would ask is refused.** Reads run,
because your permission rules already say reads run. Everything else stops with
a message naming the setting that would allow it.

To let workflows write files or run commands, say so deliberately:

```toml
[permissions.tools]
write_file = "allow"
```

That allows it everywhere — including from a webhook — which is why it is a
decision made once, in a file, and not a checkbox on the canvas.

## Schedules

A schedule step takes either an interval or a time of day:

| | |
|---|---|
| `every` | `30s`, `5m`, `2h`, `1d` |
| `at` | `09:00` — **in UTC**, not your local time |

A time of day wins if both are set. Half a minute is the shortest interval
allowed: a model step on a one-second timer is a machine kept permanently busy
by a typo.

Two things worth knowing:

- **Nothing fires until the workflow is switched to Live.** A half-drawn flow on
  a schedule would start running as it was being built.
- **Schedules are re-armed when ozgent starts.** A machine that was off
  overnight does not wake up and fire a night of missed runs at once. The first
  run after a restart is one interval later.

## Webhooks

A **when called** step has a URL:

    POST http://<this machine>:7333/hooks/<workflow>/<step>

The JSON body is what the step outputs, so `{{ hook.order_id }}` reads a field
from the caller. The reply says what happened:

```json
{ "status": "ok", "run_id": 42, "ms": 1180, "error": null }
```

It answers only while the workflow is Live. An unknown workflow, a workflow that
is switched off, and a step that is not a webhook all give the same reply —
telling them apart would let someone probe for workflows.

**There is no authentication on this URL**, exactly as there is none on the rest
of `ozgent web`. Anyone who can reach the port and knows the identifier can
start the workflow. Bind to `127.0.0.1` unless you meant otherwise.

## Runs

The last fifty runs of each workflow are kept, with every step's output, timing
and error. Older ones are dropped as new ones arrive — a workflow on a
one-minute schedule writes half a million rows a year, and the interesting ones
are always the most recent.

A failed step stops what depended on it and nothing else: an unrelated branch of
the same workflow still runs, and the run is reported as failed.

## Saving

The canvas saves as you work. A workflow is saved even when it cannot yet run —
half-drawn is the normal state of something being drawn. What being *finished*
gates is **Live**: the switch refuses to stay on while anything is unresolved,
and the header says what is wrong.

See also [tools and permissions](tools.md) and [settings](settings.md).
