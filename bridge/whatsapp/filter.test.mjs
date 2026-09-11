// Tests for the bridge's message filter.
//
// Run by `the_bridge_never_answers_itself` in ozgent-channels, which skips when
// node is not installed.

import { decide, isBookkeeping, numberOf, textOf, userOf } from "./filter.mjs";
import assert from "node:assert/strict";
import test from "node:test";

const ME = "15550000000@s.whatsapp.net";
const FRIEND = "15551111111@s.whatsapp.net";
const GROUP = "12345-67890@g.us";

const msg = (over = {}) => ({
  key: { remoteJid: FRIEND, fromMe: false, id: "A1", ...(over.key || {}) },
  message: over.message === undefined ? { conversation: "hello" } : over.message,
});

const MY_LID = "207741234567890@lid";
const self = new Set([userOf(ME), userOf(MY_LID)]);
const on = { selfChat: true, self, ourIds: new Set() };

test("a message from someone else is answered", () => {
  const out = decide(msg(), on);
  assert.equal(out.chat, FRIEND);
  assert.equal(out.own, false);
  assert.equal(out.group, false);
});

test("your own message to someone else is never answered", () => {
  // Sent from your phone into a conversation with a friend. Replying would be
  // ozgent talking over you, in your name, to them.
  const out = decide(msg({ key: { remoteJid: FRIEND, fromMe: true } }), on);
  assert.equal(out, null);
});

test("your own message to yourself is answered when the setting is on", () => {
  const out = decide(msg({ key: { remoteJid: ME, fromMe: true } }), on);
  assert.equal(out.own, true);
  assert.equal(out.chat, ME);
});

test("the self-chat is ignored entirely when the setting is off", () => {
  // Whoever uses that chat as a notepad keeps their notepad.
  const off = { selfChat: false, self, ourIds: new Set() };
  assert.equal(decide(msg({ key: { remoteJid: ME, fromMe: true } }), off), null);
});

test("ozgent never answers its own reply", () => {
  // The loop this whole module exists to prevent: in the self-chat our reply
  // comes back as another `fromMe` message in the same chat.
  const ourIds = new Set(["REPLY1"]);
  const ours = msg({ key: { remoteJid: ME, fromMe: true, id: "REPLY1" } });
  assert.equal(decide(ours, { ...on, ourIds }), null);

  // But the next thing you actually type is still answered.
  const yours = msg({ key: { remoteJid: ME, fromMe: true, id: "B2" } });
  assert.equal(decide(yours, { ...on, ourIds }).own, true);
});

test("an edit is never read as a new question", () => {
  // Streaming a reply sends one of these about once a second, each with a new
  // id that is not in the set of ids we sent — so the id check alone would not
  // catch them and ozgent would answer its own typing.
  const edit = msg({
    key: { remoteJid: ME, fromMe: true, id: "NEW" },
    message: { protocolMessage: { editedMessage: { conversation: "half a reply" } } },
  });
  assert.equal(decide(edit, on), null);
});

test("reactions, polls and key distribution are not questions", () => {
  for (const message of [
    { reactionMessage: { text: "👍" } },
    { pollUpdateMessage: {} },
    { senderKeyDistributionMessage: {} },
    { editedMessage: { conversation: "x" } },
  ]) {
    assert.equal(isBookkeeping(message), true);
    assert.equal(decide(msg({ message }), on), null);
  }
});

test("status broadcasts are not a conversation", () => {
  const out = decide(msg({ key: { remoteJid: "status@broadcast" } }), on);
  assert.equal(out, null);
});

test("a group message reports the person, not the group", () => {
  // The allowlist is about who is speaking; the group is only where.
  const out = decide(
    msg({ key: { remoteJid: GROUP, participant: FRIEND } }),
    on,
  );
  assert.equal(out.group, true);
  assert.equal(out.senderJid, FRIEND);
  assert.equal(out.chat, GROUP);
});

test("a group is never mistaken for the self-chat", () => {
  // `own` skips the allowlist, so it must not be reachable from a group even
  // if the operator is the one who sent the message.
  const out = decide(msg({ key: { remoteJid: GROUP, participant: ME, fromMe: true } }), on);
  assert.equal(out, null);
});

test("nothing at all is not a message", () => {
  assert.equal(decide(null, on), null);
  assert.equal(decide({}, on), null);
  assert.equal(decide(msg({ message: null }), on), null);
  assert.equal(decide(msg({ key: { remoteJid: "" } }), on), null);
});

test("a device suffix does not stop the self-chat matching", () => {
  // `sock.user.id` carries one and chat ids never do, which is why the bridge
  // rebuilds the JID from the number rather than comparing them raw.
  assert.equal(numberOf("15550000000:12@s.whatsapp.net"), "15550000000");
  assert.equal(numberOf("15550000000@s.whatsapp.net"), "15550000000");
  assert.equal(numberOf(null), "");
});

test("text is found wherever WhatsApp put it", () => {
  assert.equal(textOf({ conversation: "a" }), "a");
  assert.equal(textOf({ extendedTextMessage: { text: "b" } }), "b");
  assert.equal(textOf({ imageMessage: { caption: "c" } }), "c");
  assert.equal(textOf({ stickerMessage: {} }), "");
  assert.equal(textOf(null), "");
});

// ------------------------------------------------------------------- LIDs
//
// WhatsApp now addresses people by an opaque `…@lid` id. A bridge that only
// knew phone-number JIDs failed both ways at once: your own chat, arriving
// under your LID, was taken for "you, talking to someone else" and dropped;
// and a friend's number in the allowlist never matched their LID.

test("your own chat under your LID is your own chat", () => {
  const out = decide(msg({ key: { remoteJid: MY_LID, fromMe: true } }), on);
  assert.equal(out.own, true);
  assert.equal(out.chat, MY_LID, "replies go back to the chat as addressed");
});

test("a device suffix on your id does not hide it", () => {
  const out = decide(msg({ key: { remoteJid: "207741234567890:12@lid", fromMe: true } }), on);
  assert.equal(out.own, true);
});

test("the self-chat addressed by LID with the number alongside is still yours", () => {
  const out = decide(msg({ key: { remoteJid: "999@lid", remoteJidAlt: ME, fromMe: true } }), on);
  assert.equal(out.own, true);
});

test("someone writing by LID is known by their number when WhatsApp gives it", () => {
  const out = decide(msg({ key: { remoteJid: "31337@lid", remoteJidAlt: FRIEND } }), on);
  assert.equal(out.own, false);
  assert.equal(out.senderJid, "31337@lid");
  assert.equal(out.phoneJid, FRIEND);
});

test("in a group the participant's number comes from participantAlt", () => {
  const out = decide(msg({ key: { remoteJid: GROUP, participant: "31337@lid", participantAlt: FRIEND } }), on);
  assert.equal(out.group, true);
  assert.equal(out.phoneJid, FRIEND);
});

test("your own LID in someone else's chat is not the self-chat", () => {
  // fromMe into a friend's chat, even though WhatsApp knows you by LID.
  assert.equal(decide(msg({ key: { remoteJid: "31337@lid", fromMe: true } }), on), null);
});

test("Channels and broadcast lists are not conversations", () => {
  for (const remoteJid of ["120363267230866345@newsletter", "1234@broadcast", "status@broadcast"]) {
    assert.equal(decide(msg({ key: { remoteJid } }), on), null, remoteJid);
  }
});
