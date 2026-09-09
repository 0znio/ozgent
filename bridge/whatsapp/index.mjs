// Speaks WhatsApp on ozgent's behalf.
//
// stdout is the protocol — one JSON object per line, and nothing else, ever.
// Baileys and its dependencies log freely, so the first thing this file does is
// make sure none of that can reach stdout and corrupt the stream.

import { createRequire } from 'node:module'
import { mkdirSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { decide, numberOf, textOf } from './filter.mjs'

const require = createRequire(import.meta.url)

// ---------------------------------------------------------------- protocol

function emit (object) {
  process.stdout.write(JSON.stringify(object) + '\n')
}

function log (...parts) {
  process.stderr.write(parts.join(' ') + '\n')
}

// A library writing to stdout would land in the middle of a JSON line and take
// the whole channel down with it. Redirected before anything is imported.
const realStdoutWrite = process.stdout.write.bind(process.stdout)
let protocolOpen = false
process.stdout.write = (chunk, ...rest) => {
  if (protocolOpen) return realStdoutWrite(chunk, ...rest)
  return process.stderr.write(chunk, ...rest)
}
const guarded = (fn) => (...args) => {
  protocolOpen = true
  try { return fn(...args) } finally { protocolOpen = false }
}
const send = guarded(emit)

console.log = (...a) => log(...a.map(String))
console.info = console.log
console.warn = (...a) => log(...a.map(String))

// ------------------------------------------------------------------ imports

let baileys, qrcode, pino
try {
  baileys = require('@whiskeysockets/baileys')
  qrcode = require('qrcode-terminal')
  pino = require('pino')
} catch (e) {
  send({
    type: 'fatal',
    reason:
      'the WhatsApp bridge is not installed. Run `ozgent channel install whatsapp`, ' +
      'or `npm install` in the bridge directory. (' + e.message + ')'
  })
  process.exit(1)
}

// Baileys ships CommonJS, so the shape of what a default import yields differs
// between its versions and between bundlers. Every layout it has used is
// unwrapped here rather than guessed at.
const api = baileys.default && baileys.default.makeWASocket ? baileys.default : baileys
const makeWASocket =
  typeof api === 'function' ? api : (api.makeWASocket || api.default)
const { useMultiFileAuthState, DisconnectReason, downloadMediaMessage, Browsers } = api

if (typeof makeWASocket !== 'function') {
  send({ type: 'fatal', reason: 'the installed baileys does not export a socket constructor' })
  process.exit(1)
}

// -------------------------------------------------------------------- state

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
}
const stateDir = args.get('state') || process.env.OZGENT_WA_STATE
// `login` exits as soon as the device is linked; the gateway keeps running.
const once = args.has('login')
// Answer in the chat you have with yourself. Off unless asked for: plenty of
// people use that chat as a notepad, and having an assistant reply to a
// shopping list is not a feature.
const selfChat = args.get('self-chat') === '1'

if (!stateDir) {
  send({ type: 'fatal', reason: 'no --state directory given' })
  process.exit(1)
}
mkdirSync(stateDir, { recursive: true })

/** Messages this process has sent, by the token ozgent gave them. */
const sent = new Map()
/**
 * Ids of messages this process sent, so its own replies are not read back as
 * new questions. Only matters in the self-chat, where everything — ours and
 * yours alike — is `fromMe`. Bounded, because a long-running gateway would
 * otherwise grow this forever.
 */
const ourIds = new Set()
const OUR_IDS_KEPT = 500

function remember (key) {
  if (!key || !key.id) return
  ourIds.add(key.id)
  if (ourIds.size > OUR_IDS_KEPT) ourIds.delete(ourIds.values().next().value)
}

/** This account's own JID, which is also the id of the self-chat. */
let selfJid = null
let sock = null
let stopping = false

// ------------------------------------------------------------------ helpers

// A vision model resizes to a few hundred pixels, so a larger download buys
// nothing — and the sender is remote, so the size is not ours to trust.
const MAX_IMAGE_BYTES = 12 * 1024 * 1024

async function imageOf (m) {
  const image = m.message && m.message.imageMessage
  if (!image) return null
  if (image.fileLength && Number(image.fileLength) > MAX_IMAGE_BYTES) {
    log('skipping an image of', image.fileLength, 'bytes')
    return null
  }
  try {
    const buffer = await downloadMediaMessage(m, 'buffer', {})
    return { data: buffer.toString('base64'), mime: image.mimetype || 'image/jpeg' }
  } catch (e) {
    log('could not download an image:', e.message)
    return null
  }
}

// -------------------------------------------------------------------- socket

async function connect () {
  const { state, saveCreds } = await useMultiFileAuthState(stateDir)

  sock = makeWASocket({
    auth: state,
    // Deprecated in favour of handling the `qr` event, and it would print to
    // stdout — which is the protocol.
    printQRInTerminal: false,
    logger: pino({ level: 'silent' }),
    browser: Browsers ? Browsers.appropriate('ozgent') : undefined,
    // Nothing here reads anyone's history, and syncing it on every link is
    // slow and stores messages this machine was never sent.
    syncFullHistory: false,
    markOnlineOnConnect: false
  })

  sock.ev.on('creds.update', saveCreds)

  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update

    if (qr) {
      qrcode.generate(qr, { small: true }, (ascii) => {
        send({ type: 'qr', ascii })
      })
    }

    if (connection === 'open') {
      const me = sock.user && sock.user.id
      // The JID carries a device suffix (`:12`) that chat ids never have, so
      // it is rebuilt from the number to be comparable with `remoteJid`.
      selfJid = numberOf(me) ? `${numberOf(me)}@s.whatsapp.net` : null
      send({ type: 'ready', who: numberOf(me) ? '+' + numberOf(me) : 'this account' })
      if (once) {
        // Give the credentials a moment to finish being written before the
        // process ends, or the next start asks for the QR again.
        setTimeout(() => process.exit(0), 1500)
      }
      return
    }

    if (connection === 'close') {
      if (stopping) return
      const status =
        lastDisconnect && lastDisconnect.error && lastDisconnect.error.output
          ? lastDisconnect.error.output.statusCode
          : undefined

      if (status === (DisconnectReason && DisconnectReason.loggedOut)) {
        send({
          type: 'fatal',
          reason:
            'this device was unlinked from WhatsApp. Run `ozgent channel login whatsapp` to link it again.'
        })
        process.exit(1)
      }
      send({ type: 'notice', text: 'WhatsApp disconnected; reconnecting' })
      setTimeout(() => { connect().catch(fail) }, 3000)
    }
  })

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    // `append` is history being filled in behind us, not something just said.
    if (type !== 'notify') return
    for (const m of messages) {
      const verdict = decide(m, { selfChat, selfJid, ourIds })
      if (!verdict) continue
      const { chat, senderJid, group, own } = verdict

      const text = textOf(m.message)
      const image = await imageOf(m)
      if (!text.trim() && !image) continue

      send({
        type: 'message',
        chat,
        sender: numberOf(senderJid),
        jid: senderJid || '',
        name: m.pushName || numberOf(senderJid) || 'someone',
        text,
        group,
        // True only for your own chat with yourself: the account that scanned
        // the QR, which ozgent is logged in as. ozgent admits that without an
        // allowlist entry, because it is definitionally the operator.
        own,
        image
      })
    }
  })
}

function fail (e) {
  send({ type: 'fatal', reason: String((e && e.message) || e) })
  process.exit(1)
}

// ------------------------------------------------------------------ commands

async function handle (command) {
  if (!sock) return
  switch (command.type) {
    case 'typing':
      // Best-effort everywhere: presence fails on a chat we have not opened,
      // and a failed hint must not take the reply down with it.
      try {
        await sock.presenceSubscribe(command.chat)
        await sock.sendPresenceUpdate('composing', command.chat)
      } catch (e) { log('presence:', e.message) }
      break

    case 'post': {
      const result = await sock.sendMessage(command.chat, { text: command.text })
      if (result && result.key) {
        sent.set(command.token, { key: result.key, chat: command.chat })
        remember(result.key)
      }
      break
    }

    case 'revise':
    case 'settle': {
      const known = sent.get(command.token)
      if (!known) break
      try {
        const result = await sock.sendMessage(known.chat, { text: command.text, edit: known.key })
        if (result && result.key) remember(result.key)
      } catch (e) {
        // WhatsApp refuses to edit a message past about fifteen minutes, and
        // caps how many times one may be edited. A refused edit is a cosmetic
        // loss; the finished reply is posted in full regardless.
        log('edit refused:', e.message)
      }
      if (command.type === 'settle') sent.delete(command.token)
      break
    }

    case 'logout':
      stopping = true
      try { await sock.logout() } catch (e) { log('logout:', e.message) }
      process.exit(0)
      break

    default:
      log('unknown command', command.type)
  }
}

createInterface({ input: process.stdin }).on('line', (line) => {
  if (!line.trim()) return
  let command
  try {
    command = JSON.parse(line)
  } catch (e) {
    log('unreadable command:', e.message)
    return
  }
  handle(command).catch((e) => log('command failed:', e.message))
})

// ozgent closing the pipe is how the bridge is asked to stop.
process.stdin.on('close', () => process.exit(0))
process.on('uncaughtException', fail)
process.on('unhandledRejection', fail)

connect().catch(fail)
