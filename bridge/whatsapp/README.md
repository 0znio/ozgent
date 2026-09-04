# ozgent WhatsApp bridge

WhatsApp has no published protocol and no Rust client. This is a small Node
program that speaks it on ozgent's behalf, driven over stdin/stdout by
`ozgent-channels` — the same shape as the Python tool worker: a child process,
newline-delimited JSON, and nothing shared but the pipe.

    ozgent channel login whatsapp     scan the QR once
    ozgent gateway                    run it

## What it is and is not

It links your **own** WhatsApp account as a second device, the way WhatsApp Web
does. That is what makes it work with the account you already have, and it is
also its whole risk profile:

- Automating a personal account is against WhatsApp's terms of service.
  Accounts have been banned for it. Use a number you can afford to lose.
- The link is a full login. The credentials under
  `~/ozgent/channels/whatsapp/auth` can read and send your messages. Treat that
  directory like a password file; `ozgent channel logout whatsapp` removes it
  and unlinks the device.
- WhatsApp's official route for programs is the Cloud API, which is a business
  product with a webhook and an approved-template rule. It has neither of these
  risks and none of this convenience.

## Version pinning

`@whiskeysockets/baileys` is pinned to an exact version rather than a range,
and the reason is not caution in general. The registry carries a `6.17.16`
published in March 2025, which is *semver-newer* than the `6.7.24` released in
July 2026 — so `^6.7.24` silently resolves sixteen months backwards, onto a
version that no longer talks to WhatsApp. The pin is the fix.

## Protocol

One JSON object per line, both directions. stdout is the protocol; everything
human goes to stderr.

Out:

    {"type":"qr","ascii":"…"}                 scan this
    {"type":"ready","who":"+15550000000"}     linked
    {"type":"notice","text":"…"}              reconnecting, and so on
    {"type":"message","chat":"…","sender":"…","name":"…","text":"…",
     "group":false,"image":{"data":"<base64>","mime":"image/jpeg"}}
    {"type":"fatal","reason":"…"}             will not recover

In:

    {"type":"typing","chat":"…"}
    {"type":"post","token":1,"chat":"…","text":"…"}
    {"type":"revise","token":1,"text":"…"}
    {"type":"settle","token":1,"text":"…"}
    {"type":"logout"}
