# Talking to ozgent from Telegram and WhatsApp

A channel lets you message ozgent from your phone. It runs the same engine, the
same model registry, the same memory store and the same permission rules as
everything else — so a conversation started on a phone is in the web interface
when you get back to your desk, and an answer given in the browser applies to a
question asked from a chat.

    ozgent gateway              answer messages
    ozgent gateway --web        and serve the web interface, sharing one model

## Read this part first

Every other way to reach ozgent needs someone at this machine, or on a network
you chose to bind to. A bot handle is reachable by anyone in the world who types
it, and behind it sit `write_file` and `run_command`.

So two things are off until you turn them on, and neither has a convenient
default:

- **`[channels] enabled` is false.** No channel starts.
- **The allowlist is empty, which admits nobody.** Not "everyone until you
  restrict it" — nobody, until you name someone.

An allowlisted person is, for practical purposes, as trusted as someone sitting
at this keyboard. They can approve tool calls, and the approval prompt goes to
*them*, not to you. Allowlist people you would hand the laptop to.

You can narrow that. `tools` under a channel offers only the tools you name,
whatever the permission rules would otherwise allow:

```toml
[channels.telegram]
tools = ["web_search", "fetch_url", "read_file"]   # no shell, no writing
```

## Telegram

Nothing to install. Telegram is polled over an outbound HTTPS connection, so it
works from a laptop behind a router — no domain, no certificate, no port
forwarding.

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, and keep the
   token it gives you.
2. Put it in `~/ozgent/configs/config.toml`:

   ```toml
   [channels]
   enabled = true

   [channels.telegram]
   enabled = true
   token   = "123456:AA…"
   ```

   Or leave `token` out and set `$OZGENT_TELEGRAM_TOKEN`, which keeps it out of
   a file that gets copied around.
3. `ozgent gateway`. It prints a pairing code.
4. Message your bot `/pair <code>`. You are now on the allowlist, and the code
   is replaced — it works once.

You can also skip the pairing and write yourself in directly:

    ozgent channel allow telegram 4242        # by user id
    ozgent channel allow telegram @ada        # by handle

Permission questions arrive as buttons. Typing `yes`, `session`, `always` or
`no` — or `1` to `4` — does the same thing, which is often easier than scrolling
back to the message the buttons are on.

## WhatsApp

WhatsApp publishes no protocol and has no Rust client. ozgent talks to it
through a small Node program that links your own account as a second device, the
way WhatsApp Web does.

**This has real costs, and they are not hypothetical:**

- Automating a personal account is against WhatsApp's terms of service.
  Accounts have been banned for it. Use a number you can afford to lose.
- ozgent stops being a single binary. Node must be installed.
- The credentials under `~/ozgent/channels/whatsapp/auth` are a full login to
  your account. Treat that directory like a password file.

Meta's official route for programs is the Cloud API, which has none of these
problems — and cannot talk to the number you already have, needs a public
webhook, and will not let you start a conversation outside a 24-hour window
without an approved template. That trade is why this is the personal-account
route.

    ozgent channel install whatsapp     npm install, once
    ozgent channel login whatsapp       scan the QR with your phone

Then turn it on:

```toml
[channels]
enabled = true

[channels.whatsapp]
enabled = true
```

and allow yourself:

    ozgent channel allow whatsapp 15551234567

There are no buttons on WhatsApp, so a permission question arrives as numbered
options and is answered by typing a number.

Group chats are ignored unless `groups = true`. A bot that answers everything it
can see in a group is both a nuisance and a way for someone who was never
allowlisted to steer it through a member who was.

    ozgent channel logout whatsapp      unlink and forget the credentials

Remove the device from WhatsApp ▸ Linked devices as well, so the session on
their side is gone too.

## What you can type

Anything that is not one of these is a question.

| | |
|---|---|
| `/help` | what it can do, and which model is answering |
| `/new` | forget this thread and start fresh — the old one stays in the web interface |
| `/model` | which model is answering; `/model <name>` to change it |
| `/tools` | what it is allowed to use here, and what each one will ask about |
| `/stop` | stop what it is writing |
| `/whoami` | the ids an allowlist needs |
| `/pair <code>` | the only thing it answers for someone not yet allowed |

## How a reply arrives

A chat has one message that can be rewritten a limited number of times, not a
scrolling terminal. So the reply is composed and the message is edited about
once a second, with tool activity shown above the text:

```
⏳ `web_search` — running…
✓ `web_search` — 4 results · 1.2s

The three closest stations are …
```

A tool is shown from the moment the model commits to calling it, not when the
call finishes. Otherwise a model writing a file generates the entire file before
anything can be displayed, and a minute of silence on a phone reads as a dropped
connection.

When a reply outgrows one message, the current one is closed off at a sentence
or paragraph boundary and a new one continues from there — so a long answer
arrives as a sequence of complete messages, and a code block that spans the
break is closed and reopened with its language intact.

## Settings

```toml
[channels]
enabled = false
# Model for messages from a channel. Falls back to the general default_model.
# Separate because a phone is a poor place to wait on a 70B.
model = "coder"

[channels.telegram]
enabled = false
token   = ""        # or $OZGENT_TELEGRAM_TOKEN
allow   = []        # user ids or @handles; empty admits nobody, "*" admits everyone
tools   = []        # omit the key entirely to offer every tool
stream  = true      # edit one message as the reply is written

[channels.whatsapp]
enabled = false
allow   = []        # phone numbers (digits) or full JIDs
tools   = []
stream  = true
groups  = false     # answer in group chats
node    = "node"    # interpreter for the bridge
# bridge = "/path/to/bridge/whatsapp"   # found beside the executable otherwise
```

`ozgent channel status` shows all of it, plus which chats are bound to a
conversation. It never prints the token.

## When something is wrong

**Nothing is answered.** Almost always the allowlist. `ozgent channel status`
says who is on it. A message from someone not on it is logged and otherwise
ignored — deliberately, since replying to strangers confirms the bot is live.

**`another program is already polling this bot token`.** Telegram allows one
reader per token. Either another `ozgent gateway` is running, or a webhook is
set for that bot. Stop the other one, or clear the webhook.

**Telegram replies arrive without formatting.** Telegram rejects a message whose
markup it dislikes, whole, rather than stripping it — so ozgent retries as plain
text. Losing the formatting beats losing the reply. The log says which message.

**WhatsApp stops after a while.** The bridge reconnects on its own and says so.
If it says the device was unlinked, someone removed it in WhatsApp ▸ Linked
devices; run `ozgent channel login whatsapp` again.

See also [tools and permissions](tools.md) and [settings](settings.md).
