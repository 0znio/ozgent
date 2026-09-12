# Running ozgent in the background

```
ozgent daemon install     run it now, and at every login
ozgent daemon status      is it running, and what is it doing
ozgent daemon uninstall   stop it and remove the service
ozgent daemon             run it in this terminal instead
```

One process owns the model, the database, the tool worker, the messaging
channels and the scheduler, and serves the web interface and the API over all of
it. Everything else is a client of it.

## Why

Until there was a daemon, every way into ozgent brought its own everything.
`ozgent chat` loads a model, `ozgent web` loads a model, `ozgent gateway` loads a
model. Run two and there are two copies in VRAM. Run none and a job scheduled
for 9:20 does not happen, because nothing was awake to run it.

With one:

- **Scheduled jobs run** whether or not anything is open. See
  [the scheduler](scheduler.md).
- **Telegram and WhatsApp are answered** without a terminal left running.
- **The API is up** at `http://localhost:7333/v1`, OpenAI- and
  Anthropic-compatible, over the same loaded model.
- **The browser is a thin client.** So is anything else you point at the API.
- **Nothing loads a second copy of the model.**

## What it costs while idle

A daemon's real workload is the twenty-three hours a day it does nothing, and
the only honest measure is what it costs then. Three things are done about it,
in descending order of how much they matter:

**The model is not resident.** Nothing is loaded until a question arrives, and
it is dropped again after fifteen idle minutes. This is the whole ballgame: a 4B
model at Q4 is around 3 GB, and holding it overnight for one morning brief is
3 GB of nothing. The cost of getting it wrong is one reload — the same wait the
first question of the day pays anyway.

```toml
[web]
idle_unload_minutes = 15   # 0 keeps it loaded once it is in
```

Set it to `0` on a machine with VRAM to spare and a model you use all day.

**Freed memory is given back.** `free()` returns memory to the allocator, not to
the kernel — glibc keeps it in its arenas ready for the next allocation, so a
process that loads and unloads a model looks, to `ps` and to anyone watching a
service, as though it never let go. The daemon asks for the arenas to be trimmed
after every unload, so what it releases is actually released.

**It sleeps until there is something to do.** The scheduler wakes when the next
job is due rather than on a fixed tick, with a one-minute ceiling so a job
created in another process is noticed. With nothing scheduled that is one wake a
minute doing a single indexed lookup that matches nothing.

What is left is a tokio runtime, an HTTP listener, a SQLite handle, and one
polling connection per configured channel.

## Which init system

`ozgent daemon install` works out what is supervising this machine and writes
the right kind of service. systemd is not the only one, and a program that
assumes it is simply does not install on Void, Alpine, Artix or a Mac.

| | | |
|---|---|---|
| **systemd** | a user unit in `~/.config/systemd/user` | installed and started for you |
| **launchd** (macOS) | a LaunchAgent in `~/Library/LaunchAgents` | installed and started for you |
| **dinit** | `~/.config/dinit.d/ozgent` | installed and started for you |
| **OpenRC** | `/etc/init.d/ozgent` | written for you; two `sudo` lines to install |
| **runit** | `/etc/sv/ozgent/run` | written for you; two `sudo` lines to install |
| **s6** | `/etc/s6/sv/ozgent/run` | written for you; two `sudo` lines to install |

Where a per-user service is possible, that is what you get — no root, running as
the account whose `~/ozgent` this is, which is the only account that should be
reading that directory anyway. The three that have no per-user services need
root, so ozgent generates the file (the part that has to be right) and prints
the two commands rather than running `sudo` on your behalf. Those services drop
to your user with `command_user` / `chpst -u` / `s6-setuidgid`, so the daemon is
still not running as root.

`./install.sh` offers to do all of this at the end of an install.

If nothing is recognised, `ozgent daemon` in the foreground is the whole
feature — anything that keeps a command running will supervise it.

## After logging out

On systemd a *user* service stops when your last session ends, which on a server
is the moment you close SSH — exactly when a daemon is most expected to keep
going. `ozgent daemon install` says so if it applies, and the fix is:

```
sudo loginctl enable-linger $USER
```

`ozgent daemon status` shows whether lingering is on.

## Reaching it

```
http://localhost:7333            the web interface
http://localhost:7333/scheduler  scheduled jobs
http://localhost:7333/admin      gateway and model downloads (needs a password)
http://localhost:7333/v1         OpenAI- and Anthropic-compatible API
```

It binds `127.0.0.1` by default — this machine only. To reach it from a phone on
the same network, install it with `--host 0.0.0.0`, and read the warning in
[settings](settings.md) first: there is no password on anything but `/admin`.

## What it is doing

```
$ ozgent daemon status
init          systemd
service       /home/you/.config/systemd/user/ozgent.service
state         active (enabled at login)
lingering     on
idle          the model is dropped after 15 minutes
scheduler     running jobs
channels      answered here

logs          journalctl --user -u ozgent -f
```

`scheduler` and `channels` come from lock files, not from the init system: they
say whether *this* process is the one running jobs and answering Telegram, which
is a different question from whether the service is up. Only one process does
each, so a second ozgent started by hand will say `not running` there while
working perfectly well for everything else.

## The terminal, while a daemon is running

`ozgent chat` still loads its own model. On a machine with enough VRAM for two
copies that is fine; on most it is not, and the daemon will have released its
copy fifteen minutes later anyway. Use the browser or the API to talk to the
daemon.

## Stopping it

```
ozgent daemon uninstall
```

Stops it, turns it off, removes the service file. Your models, conversations,
settings and scheduled jobs are untouched — they are in `~/ozgent`, which
nothing here writes to.

See also [the scheduler](scheduler.md), [channels](channels.md) and
[the API](api.md).
