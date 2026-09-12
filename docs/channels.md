# Talking to ozgent from Telegram and WhatsApp

The gateway lets you message ozgent from your phone. It runs the same engine,
the same model registry, the same memory and the same permission rules as
everything else — so a conversation started on a phone is in the web interface
when you get back to your desk.

```
ozgent gateway telegram     set up Telegram, or change it
ozgent gateway whatsapp     link WhatsApp, or change it
ozgent gateway status       what is set up, what is running, who is allowed
```

Setting a channel up is answering a few questions. Nothing needs editing by
hand, and nothing needs a restart: `ozgent web` answers the channels while it
runs, and picks up every change within a couple of seconds — whether it was
made with these commands or on its **/admin** page.

![the gateway on the admin page](images/admin-gateway.png)

## Read this part first

Every other way to reach ozgent needs someone at this machine, or on a network
you chose. A bot handle is reachable by anyone in the world who types it, and
behind it sit `write_file` and `run_command`.

So **nobody is allowed until you name them.** An empty list admits nobody —
not everyone. Someone you allow is, for practical purposes, as trusted as
someone sitting at this keyboard: they can approve tool calls, and the
approval question goes to *them*, not to you. Allow people you would hand the
laptop to.

You can narrow that per channel, and both setups ask:

- **Tools** — all of them, only the ones you pick, or none.
- **Approvals** — whether someone in the chat may approve a tool that asks
  first (writing a file, running a command). Off, anything that would ask is
  refused, and only what your rules allow outright runs.

## Telegram

Nothing to install. Telegram is polled over an outbound connection, so it
works on a laptop behind a router — no domain, no certificate, no port
forwarding.

```
$ ozgent gateway telegram
Telegram

  1. In Telegram, open @BotFather and send /newbot. Answer its two questions.
  2. It replies with a token like 123456789:AAH… — paste it below.

  bot token (hidden): ••••••••••••••••••••••••••••••••••••••••••••••
  ✓ this is @my_ozgent_bot

Who should be able to message @my_ozgent_bot?
  1  only me
  2  me and other people
  3  only other people
> [1]

  Now, from your own Telegram, send this to @my_ozgent_bot:

      OZ4F9A1C
```

Sending that code is how ozgent learns your user id without you having to look
it up — the code is on your screen, so whoever sends it is you. (You can type
your numeric id instead; @userinfobot tells you it.) Then it asks about
tools and approvals, and you're done.

The token is checked with Telegram before anything is saved, so a typo is an
error on the spot rather than a channel that silently fails later. It is
stored in `config.toml` (readable only by you), or set
`$OZGENT_TELEGRAM_TOKEN` to keep it out of the file.

Permission questions arrive as buttons. Typing `yes`, `session`, `always` or
`no` — or `1` to `4` — does the same thing.

## WhatsApp

WhatsApp publishes no protocol and has no Rust client. ozgent talks to it
through a small Node program that links your own account as a second device,
the way WhatsApp Web does. Setup installs it for you (once, about a minute) —
you need Node 18 or newer.

**This has real costs, and they are not hypothetical:**

- Automating a personal account is against WhatsApp's terms of service.
  Accounts have been banned for it. Use a number you can afford to lose.
- The credentials under `~/ozgent/channels/whatsapp/auth` are a full login to
  your account. Treat that directory like a password file.

```
$ ozgent gateway whatsapp
  Link WhatsApp? [Y/n]
  (a QR code)
  On your phone: WhatsApp ▸ Settings ▸ Linked devices ▸ Link a device
  ✓ linked +91 98765 43210

  Which phone numbers may message it? With the country code, separated by commas.
  Your own number (+91 98765 43210) means your "Message yourself" chat.
  > +91 98765 43210
  ✓ your own "Message yourself" chat will be answered
```

On `/admin` it's the **Link WhatsApp** button: the QR code appears on the page
and follows WhatsApp as it changes the code every twenty seconds.

### Your own chat, or other people

Because the bridge links *your* account, ozgent *is* your number. The natural
place to talk to it is the chat WhatsApp gives you with yourself ("Message
yourself") — enter your own number and that is what you get. It is off unless
you ask, because plenty of people use that chat as a notepad.

Enter someone else's number and ozgent replies to them **as you**, from your
number. That is a different thing from a personal assistant; be deliberate.

Numbers can be typed any way — `+91 98765 43210`, `0091-98765-43210` — and are
stored as digits with the country code.

There are no buttons on WhatsApp, so a permission question arrives as numbered
options and is answered by typing a number. Group chats are ignored unless you
turn them on: a bot answering everything in a group is a way for someone never
allowed to steer it through a member who was.

## Changing things later

Run the same command again and you get a menu:

```
$ ozgent gateway telegram
telegram · @my_ozgent_bot · on · 2 allowed
  1  who may message it
  2  allow someone
  3  remove someone
  4  tools                 (web_search, fetch_url)
  5  approve tool calls    (yes)
  6  change the bot token
  7  turn it off
  8  sign out
  9  done
```

Or say it directly — handy in scripts:

| | |
|---|---|
| `ozgent gateway telegram allow 4242 @ada_l` | let people in (ids, @usernames) |
| `ozgent gateway whatsapp allow "+1 555 123 4567"` | a number, with its country code |
| `ozgent gateway <channel> deny <who>` | take someone off the list |
| `ozgent gateway <channel> allowed` | who is on it |
| `ozgent gateway <channel> tools web_search,fetch_url` | only these; also `all`, `none` |
| `ozgent gateway telegram token` | a new bot token (asked for, hidden) |
| `ozgent gateway whatsapp link` | link a different account |
| `ozgent gateway <channel> signout` | forget the token / unlink the device |
| `ozgent gateway <channel> on` / `off` | stop answering, keep the settings |

Signing WhatsApp out removes the device from your phone's Linked devices list
too. Signing Telegram out forgets the token here; revoke it with @BotFather
(`/revoke`) if it should stop working everywhere.

### Letting someone in from their phone

When the gateway runs it has a **pairing code** — shown on `/admin` and when
`ozgent gateway` starts. Someone not on the list can send `/pair CODE` to the
bot and be added. The code works once and changes after it is used. It is the
only thing ozgent answers for someone not on the list; everyone else is
ignored, deliberately, since replying to strangers confirms the bot is live.

## Running it

The best answer is [the daemon](daemon.md) — one process in the background that
answers the channels whether or not anything is open:

```
ozgent daemon install
```

`ozgent web` answers them too, while it runs. To answer them from a terminal
without the web interface:

```
ozgent gateway              answer messages from this terminal
ozgent gateway --web        and serve the web interface, sharing one model
```

Only one ozgent can answer a channel at a time — Telegram allows one reader per
bot token, and a second WhatsApp connection replaces the first. The second one
to start says who has the channels, and takes over if that one stops.

## What you can type

Anything that is not one of these is a question.

| | |
|---|---|
| `/help` | what it can do, and which model is answering |
| `/new` | forget this thread and start fresh — the old one stays in the web interface |
| `/model` | which model is answering; `/model <name>` to change it |
| `/tools` | what it is allowed to use here, and what each one will ask about |
| `/stop` | stop what it is writing |
| `/whoami` | the ids a list needs |
| `/pair <code>` | the only thing it answers for someone not yet allowed |

`@agent` works here too, and the model can hand a question to an agent itself.

You can also ask for something on a timer — *"every weekday at 9:20, send me a
pre-market brief"* — and the answers arrive in this chat. A job set up from
the terminal or the page with no chat named goes to everyone on this channel's
list, re-checked each time it sends, so removing someone stops their
deliveries. A job created from a
chat may only answer back into that same chat, and gets that channel's tools
and no more; otherwise "schedule me a reminder" would be a way to make ozgent
message somebody else, or to reach a tool the channel does not allow. See
[the scheduler](scheduler.md).

## How a reply arrives

A chat has one message that can be rewritten a limited number of times, not a
scrolling terminal. So the reply is composed and the message is edited about
once a second, with tool activity shown above the text:

```
⏳ `web_search` — running…
✓ `web_search` — 4 results · 1.2s

The three closest stations are …
```

When a reply outgrows one message, the current one is closed off at a sentence
or paragraph boundary and a new one continues from there, and a code block
that spans the break is closed and reopened with its language intact.

## Settings

Everything above is stored in `~/ozgent/configs/config.toml`; you never need
to touch it, but this is what it looks like:

```toml
[channels]
enabled = true
model = "coder"      # optional: the model chats get. Falls back to default_model.

[channels.telegram]
enabled = true
token   = "…"        # or $OZGENT_TELEGRAM_TOKEN
allow   = ["4242", "@ada_l"]
tools   = ["web_search", "fetch_url"]   # leave out for every tool; [] for none
approve = true       # may a tool that asks be approved from the chat
stream  = true       # edit one message as the reply is written

[channels.whatsapp]
enabled   = true
allow     = ["15551234567"]
self_chat = true     # answer your own "Message yourself" chat
groups    = false
approve   = false
node      = "node"
```

## When something is wrong

**Nothing is answered.** Almost always the list. `ozgent gateway status` says
who is on it. A message from someone not on it is logged and otherwise ignored.

**`another program is already polling this bot token`.** Telegram allows one
reader per token. Either another ozgent is running, or a webhook is set for
that bot. Stop the other one, or clear the webhook.

**WhatsApp says it is not linked any more.** Someone removed the device in
WhatsApp ▸ Linked devices. Run `ozgent gateway whatsapp link`, or press Link
on `/admin`.

**Telegram replies arrive without formatting.** Telegram rejects a message
whose markup it dislikes, whole, so ozgent retries as plain text. Losing the
formatting beats losing the reply.

See also [tools and permissions](tools.md) and [settings](settings.md).
