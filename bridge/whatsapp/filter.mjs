// Deciding what counts as something a person said.
//
// Its own module, and tested on its own, because this is where the one bug
// that cannot be shrugged off lives: in the chat you have with yourself,
// *everything* is `fromMe` — your questions and ozgent's replies alike. Get the
// test wrong and ozgent answers its own answer, forever, on your real
// WhatsApp account.
//
// index.mjs holds the connection and the state; everything here is a pure
// function of a message and what we know.

/** The phone number behind a JID, digits only. */
export function numberOf (jid) {
  if (!jid) return ''
  return String(jid).split('@')[0].split(':')[0].replace(/[^0-9]/g, '')
}

/** Whatever text a message carries, wherever WhatsApp put it. */
export function textOf (message) {
  if (!message) return ''
  return (
    message.conversation ||
    (message.extendedTextMessage && message.extendedTextMessage.text) ||
    (message.imageMessage && message.imageMessage.caption) ||
    (message.videoMessage && message.videoMessage.caption) ||
    (message.documentMessage && message.documentMessage.caption) ||
    ''
  )
}

/**
 * Whether a message is bookkeeping rather than something someone said.
 *
 * An edit arrives as a protocol message with a *new* id, so it is not in the
 * set of ids we sent — and streaming a reply sends one of these about once a
 * second. Without this check ozgent would read its own edits as new questions.
 */
export function isBookkeeping (message) {
  return !!(
    message &&
    (message.protocolMessage || message.editedMessage || message.reactionMessage ||
     message.pollUpdateMessage || message.senderKeyDistributionMessage)
  )
}

/** The user part of a JID, without device suffix: `123:4@lid` → `123`. */
export function userOf (jid) {
  if (!jid) return ''
  return String(jid).split('@')[0].split(':')[0]
}

/** A phone-number JID, as opposed to a LID or a group. */
export function isPhoneJid (jid) {
  return !!jid && /@(s\.whatsapp\.net|c\.us)$/.test(String(jid))
}

/**
 * What to do with one incoming message.
 *
 * Returns `null` to ignore it, or `{ chat, senderJid, phoneJid, group, own }`.
 * `own` means the self-chat: the account ozgent is logged in as, talking to
 * itself, which is the one sender admitted without an allowlist entry.
 *
 * WhatsApp now addresses people by a LID — an opaque `…@lid` id — rather than
 * by phone number, and the self-chat may arrive under your LID as well. So
 * `self` is every id this account goes by (phone number and LID, user parts
 * only), and `phoneJid` is the sender's number when WhatsApp said it
 * (`remoteJidAlt` / `participantAlt`), for an allowlist written as numbers.
 */
export function decide (m, { selfChat = false, self = new Set(), ourIds = new Set() } = {}) {
  if (!m || !m.message || !m.key) return null
  if (isBookkeeping(m.message)) return null

  const chat = m.key.remoteJid || ''
  if (!chat) return null
  // Status updates, broadcast lists and Channels are one-to-many feeds, not
  // a conversation anyone is having with ozgent.
  if (chat === 'status@broadcast' || chat.endsWith('@broadcast') || chat.endsWith('@newsletter')) {
    return null
  }

  const group = chat.endsWith('@g.us')
  const alt = m.key.remoteJidAlt || ''
  const own = !!(
    selfChat && !group && self.size &&
    (self.has(userOf(chat)) || (alt && self.has(userOf(alt))))
  )

  if (m.key.fromMe) {
    // Sent from another of your devices, into a conversation with someone
    // else. Answering it would be ozgent talking over you.
    if (!own) return null
    // Our own reply, coming back. This is the loop.
    if (ourIds.has(m.key.id)) return null
  }

  const senderJid = group ? (m.key.participant || '') : chat
  const senderAlt = group ? (m.key.participantAlt || '') : alt
  const phoneJid = isPhoneJid(senderJid) ? senderJid : (isPhoneJid(senderAlt) ? senderAlt : '')
  return { chat, senderJid, phoneJid, group, own }
}
