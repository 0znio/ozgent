// ozgent /admin — the gateway, model downloads and the admin password.
//
// Every request that changes something carries `x-ozgent-admin`, which a
// form on another site cannot add; the session itself is an HttpOnly cookie
// this script never sees.

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "content-type": "application/json", "x-ozgent-admin": "1", ...(options.headers ?? {}) },
  });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (res.status === 401 && !path.startsWith("/api/admin/login")) {
    // Signed out underneath us: the password was changed or reset.
    showGate();
    throw new Error("signed out");
  }
  if (!res.ok) throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
  return body;
}

// ------------------------------------------------------------------ format

function bytesText(n) {
  if (!n && n !== 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

function countText(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n);
}

function durationText(seconds) {
  if (!isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ------------------------------------------------------------------ session

function showGate(state) {
  $("shell").hidden = true;
  $("gate").hidden = false;
  stopPolling();
  const configured = state?.configured ?? true;
  $("signin").hidden = !configured;
  $("forgot").hidden = !configured;
  $("not-set").hidden = configured;
  if (state?.invalid) {
    $("not-set").querySelector("p").textContent =
      "The password in config.toml is not a password hash, so it is refused. On the machine, run:";
    $("not-set").querySelector("pre").textContent = "ozgent admin reset";
  }
  if (configured) $("password").focus();
}

function showShell() {
  $("gate").hidden = true;
  $("shell").hidden = false;
  openView(location.hash.replace("#", "") || "gateway");
}

async function boot() {
  const s = await api("/api/admin/session");
  if (s.signed_in) showShell();
  else showGate(s);
}

$("signin").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("signin-btn");
  const err = $("signin-error");
  err.hidden = true;
  btn.disabled = true;
  btn.textContent = "Checking…";
  try {
    await api("/api/admin/login", { method: "POST", body: JSON.stringify({ password: $("password").value }) });
    $("password").value = "";
    showShell();
  } catch (x) {
    err.textContent = x.message;
    err.hidden = false;
    $("password").select();
  } finally {
    btn.disabled = false;
    btn.textContent = "Sign in";
  }
});

$("signout").addEventListener("click", async () => {
  await fetch("/api/admin/session", { method: "DELETE" });
  showGate({ configured: true });
});

// ------------------------------------------------------------------ views

let timers = { gateway: null, jobs: null, mcp: null };
function stopPolling() {
  clearTimeout(timers.gateway);
  clearTimeout(timers.jobs);
  clearTimeout(timers.mcp);
}

function openView(name) {
  if (!["gateway", "models", "mcp", "security", "account"].includes(name)) name = "gateway";
  for (const t of document.querySelectorAll(".adm-nav .tab")) {
    t.setAttribute("aria-selected", String(t.dataset.view === name));
  }
  for (const v of document.querySelectorAll(".adm-view")) v.hidden = v.id !== `view-${name}`;
  history.replaceState(null, "", `#${name}`);
  stopPolling();
  if (name === "gateway") pollGateway();
  if (name === "models") { openModels(); openEmbedding(); openServer(); }
  if (name === "security") openSecurity();
  if (name === "mcp") loadMcp();
}

for (const t of document.querySelectorAll(".adm-nav .tab")) {
  t.addEventListener("click", () => openView(t.dataset.view));
}

// ================================================================== gateway

const gw = {
  data: null,
  /// Which channel's token form is open for changing an existing token.
  changingToken: false,
  /// When the current QR code session started, for the countdown bar.
  linkStarted: null,
  /// Lists being edited are not re-rendered under the cursor.
  busy: new Set(),
};

const LINK_SECONDS = 180;

const PHASE = {
  off: ["off", ""],
  installing: ["installing", "live"],
  starting: ["connecting", "live"],
  linking: ["waiting for scan", "live"],
  connected: ["connected", "good"],
  failed: ["failed", "bad"],
  elsewhere: ["another ozgent", "faint"],
};

async function pollGateway() {
  clearTimeout(timers.gateway);
  try {
    gw.data = await api("/api/admin/gateway");
    renderGateway();
  } catch (e) {
    if (e.message !== "signed out") {
      const b = $("gw-banner");
      b.hidden = false;
      b.className = "adm-banner bad";
      b.textContent = e.message;
    }
  }
  const live = ["telegram", "whatsapp"].some((k) =>
    ["installing", "starting", "linking"].includes(gw.data?.[k]?.runtime?.phase));
  timers.gateway = setTimeout(pollGateway, live ? 1200 : 4000);
}

function renderGateway() {
  const d = gw.data;
  const banner = $("gw-banner");
  banner.hidden = true;
  if (!d.running_here) {
    banner.hidden = false;
    banner.className = "adm-banner";
    banner.textContent = "The gateway is not running in this ozgent, so changes are saved but nothing is answered from here.";
  } else if (!d.hosted) {
    banner.hidden = false;
    banner.className = "adm-banner";
    banner.textContent = `${d.elsewhere ?? "Another ozgent"} is answering the channels. Changes made here are saved and it picks them up; linking and signing out need that one.`;
  } else if (!d.tools_enabled) {
    banner.hidden = false;
    banner.className = "adm-banner";
    banner.textContent = "Tools are switched off in Settings, so chats can only talk.";
  }

  $("gw-pairing").textContent = d.pairing ?? "—";
  $("gw-new-code").hidden = !d.pairing;

  const sel = $("gw-model");
  if (!gw.busy.has("model")) {
    sel.replaceChildren();
    const dflt = el("option", null, d.default_model ? `the default model (${d.default_model})` : "the default model (none set)");
    dflt.value = "";
    sel.append(dflt);
    for (const m of d.models) {
      const o = el("option", null, m);
      o.value = m;
      sel.append(o);
    }
    sel.value = d.model ?? "";
  }

  renderChannel("telegram", d.telegram);
  renderChannel("whatsapp", d.whatsapp);
}

function renderChannel(kind, c) {
  const card = $(`ch-${kind}`);
  const q = (role) => card.querySelector(`[data-role="${role}"]`);
  const rt = c.runtime;
  const [label, tone] = PHASE[rt.phase] ?? [rt.phase, ""];
  const pill = q("pill");
  pill.textContent = rt.phase === "connected" && rt.who ? `connected · ${rt.who}` : label;
  pill.className = `adm-pill ${tone}`;

  const setUp = kind === "telegram" ? c.token_set : rt.linked;
  const toggle = q("enabled");
  toggle.checked = c.enabled && gw.data.enabled;
  toggle.disabled = !setUp;
  toggle.parentElement.dataset.tip = setUp ? `Answer messages on ${kind === "telegram" ? "Telegram" : "WhatsApp"}` : "Set it up first";

  const detail = q("detail");
  const showDetail = rt.detail && (rt.phase === "failed" || rt.phase === "installing" || rt.phase === "starting");
  detail.hidden = !showDetail;
  detail.textContent = rt.detail ?? "";
  detail.className = `adm-detail ${rt.phase === "failed" ? "bad" : ""}`;

  // Account row.
  const account = q("account");
  account.replaceChildren();
  if (kind === "telegram") {
    const form = q("token-form");
    const editing = !c.token_set || gw.changingToken;
    form.hidden = !editing;
    q("token-cancel").hidden = !c.token_set;
    if (c.token_set) {
      account.append(el("span", "adm-mono", rt.who ?? (c.token_from_env ? "token from $OZGENT_TELEGRAM_TOKEN" : "token set")));
    } else {
      account.append(el("span", "hint inline", "not set up"));
    }
  } else {
    if (rt.phase === "installing") {
      const box = el("div", "adm-progress");
      const bar = el("div", "adm-bar indeterminate");
      bar.append(el("span"));
      box.append(bar, el("span", "hint inline", "Installing the WhatsApp bridge with npm. This happens once and takes about a minute."));
      account.append(box);
    } else if (rt.linked) {
      account.append(el("span", "adm-mono", rt.who ?? "linked"));
    } else if (rt.phase !== "linking" && rt.phase !== "starting") {
      const b = el("button", "primary-btn", "Link WhatsApp");
      b.type = "button";
      b.disabled = !gw.data.hosted;
      b.addEventListener("click", () => linkWhatsApp(b));
      account.append(b);
      if (rt.installed === false) account.append(el("span", "hint inline", " the bridge installs itself first (needs Node.js)"));
    } else {
      account.append(el("span", "hint inline", "linking…"));
    }
    renderLinking(card, rt);
  }

  // The rest only once it is set up, so a fresh page is one clear step.
  const box = q("configured");
  if (!setUp) {
    box.replaceChildren();
    box.dataset.built = "";
    return;
  }
  if (!box.dataset.built) {
    box.replaceChildren($("tpl-configured").content.cloneNode(true));
    box.dataset.built = "1";
    wireConfigured(kind, box);
  }
  fillConfigured(kind, box, c);
}

function renderLinking(card, rt) {
  const box = card.querySelector('[data-role="link"]');
  const linking = rt.phase === "linking" || (rt.phase === "starting" && !rt.linked);
  box.hidden = !linking;
  if (!linking) { gw.linkStarted = null; return; }
  gw.linkStarted ??= Date.now();
  const img = card.querySelector('[data-role="qr"]');
  const wait = card.querySelector('[data-role="qr-wait"]');
  if (rt.phase === "linking") {
    // A new URL each poll, so the browser fetches the code WhatsApp just
    // rotated to rather than showing the one it cached.
    img.src = `/api/admin/gateway/qr?t=${Date.now()}`;
    img.hidden = false;
    wait.hidden = true;
  } else {
    img.hidden = true;
    wait.hidden = false;
  }
  // The server's count, so a reload shows the time really left.
  const left = rt.link_left ?? Math.max(0, LINK_SECONDS - (Date.now() - gw.linkStarted) / 1000);
  card.querySelector('[data-role="qr-left-bar"]').style.width = `${(left / LINK_SECONDS) * 100}%`;
  card.querySelector('[data-role="qr-left"]').textContent = `${Math.ceil(left)} s left`;
}

async function linkWhatsApp(button) {
  button.disabled = true;
  button.textContent = "Starting…";
  gw.linkStarted = Date.now();
  try {
    await api("/api/admin/gateway/whatsapp/link", { method: "POST" });
  } catch (e) {
    alertIn("whatsapp", e.message);
  }
  pollGateway();
}

function alertIn(kind, message) {
  const d = $(`ch-${kind}`).querySelector('[data-role="detail"]');
  d.hidden = false;
  d.className = "adm-detail bad";
  d.textContent = message;
}

async function putChannel(kind, body) {
  await api(`/api/admin/gateway/${kind}`, { method: "PUT", body: JSON.stringify(body) });
  pollGateway();
}

function wireConfigured(kind, box) {
  const q = (role) => box.querySelector(`[data-role="${role}"]`);
  const input = q("allow-input");
  input.placeholder = kind === "telegram" ? "user id or @username" : "phone number with country code, e.g. +91 98765 43210";
  input.addEventListener("focus", () => gw.busy.add(`allow-${kind}`));
  input.addEventListener("blur", () => gw.busy.delete(`allow-${kind}`));

  q("allow-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = q("allow-error");
    err.hidden = true;
    const add = input.value.split(",").map((s) => s.trim()).filter(Boolean);
    if (!add.length) return;
    try {
      await putChannel(kind, { allow: [...gw.data[kind].allow, ...add] });
      input.value = "";
    } catch (x) {
      err.textContent = x.message;
      err.hidden = false;
    }
  });

  q("self-row").hidden = kind !== "whatsapp";
  q("groups-row").hidden = kind !== "whatsapp";
  q("self-chat").addEventListener("change", (e) => putChannel(kind, { self_chat: e.target.checked }));
  q("groups").addEventListener("change", (e) => putChannel(kind, { groups: e.target.checked }));
  q("approve").addEventListener("change", (e) => putChannel(kind, { approve: e.target.checked }));
  q("reply-unauthorized").addEventListener("change", (e) => putChannel(kind, { reply_unauthorized: e.target.checked }));

  for (const b of q("tools-mode").querySelectorAll("button")) {
    b.addEventListener("click", () => {
      const mode = b.dataset.mode;
      const current = gw.data[kind].tools;
      const tools = mode === "all" ? null : mode === "none" ? [] : (current && current.length ? current : ["web_search", "fetch_url"].filter((t) => gw.data.tools.some((x) => x.name === t)));
      putChannel(kind, { tools });
    });
  }

  q("restart").addEventListener("click", async () => {
    await api(`/api/admin/gateway/${kind}/restart`, { method: "POST" });
    pollGateway();
  });
  q("change").addEventListener("click", () => {
    if (kind === "telegram") {
      gw.changingToken = true;
      renderGateway();
      $(`ch-telegram`).querySelector('[data-role="token"]').focus();
    } else {
      confirmInline(q("change"), "Sign this account out and link another?", async () => {
        await api("/api/admin/gateway/whatsapp/signout", { method: "POST" });
        const b = el("button");
        await linkWhatsApp(b);
      });
    }
  });
  q("signout").addEventListener("click", () => {
    const what = kind === "telegram"
      ? "Forget the bot token? Revoke it with @BotFather too if it should stop working."
      : "Unlink WhatsApp? The device is removed from your phone's Linked devices.";
    confirmInline(q("signout"), what, async () => {
      q("signout").textContent = "Signing out…";
      try {
        await api(`/api/admin/gateway/${kind}/signout`, { method: "POST" });
      } catch (x) {
        alertIn(kind, x.message);
      }
      pollGateway();
    });
  });
}

/// Ask on the spot: a second click, next to the button, says what happens.
function confirmInline(button, question, action) {
  const bar = button.parentElement;
  const old = [...bar.childNodes];
  const ask = el("span", "md-ask", question);
  const yes = el("button", "ghost-btn auto md-danger", "Yes");
  const no = el("button", "ghost-btn auto", "Cancel");
  yes.type = no.type = "button";
  const restore = () => bar.replaceChildren(...old);
  no.addEventListener("click", restore);
  yes.addEventListener("click", async () => { restore(); await action(); });
  bar.replaceChildren(ask, no, yes);
}

function fillConfigured(kind, box, c) {
  const q = (role) => box.querySelector(`[data-role="${role}"]`);
  q("self-chat").checked = !!c.self_chat;
  q("groups").checked = !!c.groups;
  q("approve").checked = !!c.approve;
  q("reply-unauthorized").checked = !!c.reply_unauthorized;

  const list = q("allow");
  list.replaceChildren();
  for (const id of c.allow) {
    const li = el("li", `adm-chip-item${id === "*" ? " bad" : ""}`);
    li.append(el("span", "adm-mono", id === "*" ? "* everyone" : kind === "whatsapp" && /^\d+$/.test(id) ? `+${id}` : id));
    const x = el("button", "adm-chip-x", "×");
    x.type = "button";
    x.setAttribute("aria-label", `Remove ${id}`);
    x.addEventListener("click", () => putChannel(kind, { allow: c.allow.filter((a) => a !== id) }));
    li.append(x);
    list.append(li);
  }
  const nobody = !c.allow.length && !(kind === "whatsapp" && c.self_chat);
  q("allow-hint").textContent = nobody
    ? "Nobody yet, so nothing is answered."
    : kind === "whatsapp" && c.allow.length
      ? "ozgent replies to these people as you, from your number."
      : "";

  const mode = c.tools === null ? "all" : c.tools.length ? "some" : "none";
  for (const b of q("tools-mode").querySelectorAll("button")) {
    b.setAttribute("aria-checked", String(b.dataset.mode === mode));
  }
  const tools = q("tools");
  tools.replaceChildren();
  tools.hidden = mode !== "some";
  for (const t of gw.data.tools) {
    const row = el("label", "adm-tool");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = (c.tools ?? []).includes(t.name);
    cb.addEventListener("change", () => {
      const now = new Set(c.tools ?? []);
      if (cb.checked) now.add(t.name); else now.delete(t.name);
      putChannel(kind, { tools: [...now] });
    });
    row.append(cb, el("span", "adm-mono", t.name), el("span", `md-chip ${t.rule === "allow" ? "md-good" : t.rule === "deny" ? "md-warn" : ""}`, t.rule));
    tools.append(row);
  }
  q("tools-hint").textContent =
    mode === "all" ? "Every tool, each following your permission rules: reading runs, writing and commands ask."
      : mode === "none" ? "It only chats."
        : "Tools not ticked are never offered here.";

  q("change").textContent = kind === "telegram" ? "Change bot token" : "Link a different account";
  q("restart").hidden = c.runtime.phase !== "failed" && c.runtime.phase !== "connected";
}

// Telegram token form.
$("ch-telegram").querySelector('[data-role="token-form"]').addEventListener("submit", async (e) => {
  e.preventDefault();
  const card = $("ch-telegram");
  const q = (role) => card.querySelector(`[data-role="${role}"]`);
  const err = q("token-error");
  const btn = q("token-save");
  err.hidden = true;
  btn.disabled = true;
  btn.textContent = "Checking with Telegram…";
  try {
    const out = await api("/api/admin/gateway/telegram/token", { method: "POST", body: JSON.stringify({ token: q("token").value }) });
    q("token").value = "";
    gw.changingToken = false;
    const d = q("detail");
    d.hidden = false;
    d.className = "adm-detail good";
    d.textContent = `Connected to ${out.bot}. Now allow yourself below — your user id, or send /pair and the code above to the bot.`;
  } catch (x) {
    err.textContent = x.message;
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Connect";
    pollGateway();
  }
});
$("ch-telegram").querySelector('[data-role="token-cancel"]').addEventListener("click", () => {
  gw.changingToken = false;
  renderGateway();
});

for (const kind of ["telegram", "whatsapp"]) {
  $(`ch-${kind}`).querySelector('[data-role="enabled"]').addEventListener("change", (e) => {
    putChannel(kind, { enabled: e.target.checked }).catch((x) => alertIn(kind, x.message));
  });
}

$("ch-whatsapp").querySelector('[data-role="link-cancel"]').addEventListener("click", async (e) => {
  e.target.disabled = true;
  try {
    await api("/api/admin/gateway/whatsapp/signout", { method: "POST" });
  } catch (x) {
    alertIn("whatsapp", x.message);
  }
  e.target.disabled = false;
  pollGateway();
});

$("gw-new-code").addEventListener("click", async () => {
  const out = await api("/api/admin/gateway/pairing", { method: "POST" });
  $("gw-pairing").textContent = out.pairing;
});

$("gw-model").addEventListener("focus", () => gw.busy.add("model"));
$("gw-model").addEventListener("blur", () => gw.busy.delete("model"));
$("gw-model").addEventListener("change", async (e) => {
  await api("/api/admin/gateway", { method: "PUT", body: JSON.stringify({ model: e.target.value || null }) });
  gw.busy.delete("model");
  pollGateway();
});

// =================================================================== models

const hub = { installed: [], jobs: [], repo: null, chosen: null, searchTimer: null, running: new Set() };

function asRepo(text) {
  const t = text.trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//, "").replace(/\/+$/, "");
  return /^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/.test(t) ? t : null;
}

async function openModels() {
  hub.installed = await api("/api/models");
  renderInstalled();
  await pollJobs();
}

async function searchHub(query) {
  const box = $("md-results");
  const note = $("md-search-note");
  const direct = asRepo(query);
  $("md-repo").hidden = true;
  box.hidden = false;
  box.replaceChildren();
  if (direct) box.append(resultRow({ id: direct.split(":")[0], downloads: null, vision: false }, "open"));
  if (query.trim().length < 2) { note.textContent = "GGUF repositories only — those are what llama.cpp runs."; return; }
  note.textContent = "searching…";
  try {
    const data = await api(`/api/hub/search?q=${encodeURIComponent(query.trim())}`);
    if ($("md-query").value.trim() !== query.trim()) return;
    for (const r of data.results) {
      if (direct && r.id.toLowerCase() === direct.split(":")[0].toLowerCase()) continue;
      box.append(resultRow(r));
    }
    note.textContent = data.results.length
      ? `${data.results.length} repositories, most downloaded first`
      : "nothing found. Try the model's family name, like qwen3.5 or gemma";
  } catch (e) {
    note.textContent = e.message;
  }
}

function resultRow(r, action) {
  const row = el("button", "md-result");
  row.type = "button";
  row.append(el("span", "md-result-id", r.id));
  const chips = el("span", "md-chips");
  if (r.vision) chips.append(el("span", "md-chip", "vision"));
  row.append(chips, el("span", "md-result-meta", action === "open" ? "open this repository" : `${countText(r.downloads ?? 0)} downloads`));
  row.addEventListener("click", () => openRepo(r.id));
  return row;
}

async function openRepo(id) {
  const panel = $("md-repo");
  const error = $("md-repo-error");
  $("md-results").hidden = true;
  panel.hidden = false;
  error.hidden = true;
  $("md-repo-id").textContent = id;
  $("md-quants").innerHTML = '<div class="adm-progress"><div class="adm-bar indeterminate"><span></span></div><span class="hint inline">reading the repository and each file\'s header…</span></div>';
  $("md-pull").disabled = true;
  try {
    const repo = await api(`/api/hub/repo?repo=${encodeURIComponent(id)}`);
    hub.repo = repo;
    $("md-repo-id").textContent = repo.id;
    $("md-repo-vision").hidden = !repo.vision;
    $("md-repo-gated").hidden = !repo.gated;
    $("md-gpu").textContent = repo.gpu
      ? `Judged against ${repo.gpu.name} (${bytesText(repo.gpu.memory)}). The context shown is what fits beside the weights on the GPU` +
        (repo.vision ? `, after the ${bytesText(repo.projector_bytes)} vision projector.` : ".")
      : "No GPU found: every size runs on the CPU, and smaller is faster.";
    renderQuants();
    if (!repo.runnable) {
      error.textContent = "This repository has no GGUF files, so there is nothing llama.cpp can run.";
      error.hidden = false;
    }
  } catch (e) {
    $("md-quants").replaceChildren();
    error.textContent = e.message;
    error.hidden = false;
  }
}

/// What a quantisation leaves room for, in words.
function fitText(q) {
  if (q.fits === false) return { text: "partly on CPU", cls: "md-faint" };
  if (q.context_fits != null) {
    const ctx = q.context_fits;
    if (ctx <= 0) return { text: "no room for context", cls: "md-warn" };
    return { text: `${countText(ctx)} context fits`, cls: ctx >= 32768 ? "md-good" : "" };
  }
  return null;
}

function renderQuants() {
  const box = $("md-quants");
  box.replaceChildren();
  const have = new Set(hub.installed.map((m) => m.reference.toLowerCase()));
  const pick = hub.repo.quants.find((q) => q.recommended) ?? hub.repo.quants[0];
  hub.chosen = pick?.quant ?? null;
  for (const q of hub.repo.quants) {
    const row = el("button", "md-quant");
    row.type = "button";
    row.setAttribute("role", "radio");
    row.append(el("span", "md-quant-name", q.quant), el("span", "md-quant-size", bytesText(q.bytes) + (q.shards > 1 ? ` · ${q.shards} files` : "")));
    const chips = el("span", "md-chips");
    if (q.recommended) chips.append(el("span", "md-chip md-good", "recommended"));
    const fit = fitText(q);
    if (fit) chips.append(el("span", `md-chip ${fit.cls}`, fit.text));
    if (have.has(q.name.toLowerCase())) chips.append(el("span", "md-chip", "installed"));
    row.append(chips);
    row.dataset.quant = q.quant;
    row.addEventListener("click", () => { hub.chosen = q.quant; syncQuants(); });
    box.append(row);
  }
  syncQuants();
  box.querySelector('[aria-checked="true"]')?.scrollIntoView({ block: "center" });
}

function syncQuants() {
  for (const row of $("md-quants").querySelectorAll(".md-quant")) {
    row.setAttribute("aria-checked", String(row.dataset.quant === hub.chosen));
  }
  const q = hub.repo?.quants.find((x) => x.quant === hub.chosen);
  const pull = $("md-pull");
  const installed = q && hub.installed.some((m) => m.reference.toLowerCase() === q.name.toLowerCase());
  const downloading = q && hub.jobs.some((j) =>
    (j.status === "resolving" || j.status === "downloading") &&
    j.repo.toLowerCase() === hub.repo.id.toLowerCase() &&
    (j.quant ?? "").toUpperCase() === q.quant.toUpperCase());
  pull.disabled = !q || installed || downloading;
  pull.textContent = !q ? "Download"
    : installed ? `${q.quant} is installed`
      : downloading ? `${q.quant} is downloading`
        : `Download ${q.quant} · ${bytesText(q.bytes + (hub.repo.projector_bytes ?? 0))}`;
}

async function startPull(repo, quant, alias) {
  const error = $("md-repo-error");
  error.hidden = true;
  try {
    await api("/api/hub/pull", { method: "POST", body: JSON.stringify({ repo, quant, alias }) });
    $("md-alias").value = "";
    await pollJobs();
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  }
}

async function pollJobs() {
  clearTimeout(timers.jobs);
  try { hub.jobs = await api("/api/hub/pulls"); } catch { hub.jobs = []; }
  const active = hub.jobs.filter((j) => j.status === "resolving" || j.status === "downloading");
  const finished = [...hub.running].filter((id) => !active.some((j) => j.id === id));
  hub.running = new Set(active.map((j) => j.id));
  if (finished.length) {
    hub.installed = await api("/api/models");
    renderInstalled();
    if (hub.repo) renderQuants();
  }
  renderJobs();
  if (hub.repo && !$("md-repo").hidden) syncQuants();
  if (!$("view-models").hidden) timers.jobs = setTimeout(pollJobs, active.length ? 700 : 3000);
}

function renderJobs() {
  const box = $("md-jobs");
  $("md-jobs-section").hidden = !hub.jobs.length;
  box.replaceChildren();
  for (const j of hub.jobs) {
    const row = el("div", `md-job ${j.status}`);
    const share = j.total ? Math.min(1, j.done / j.total) : 0;
    const running = j.status === "resolving" || j.status === "downloading";
    let line;
    if (j.status === "resolving") line = "reading the repository…";
    else if (j.status === "downloading") {
      const left = j.bytes_per_second > 0 ? durationText((j.total - j.done) / j.bytes_per_second) : "";
      line = [`${bytesText(j.done)} of ${bytesText(j.total)}`, j.bytes_per_second > 0 ? `${bytesText(j.bytes_per_second)}/s` : "starting",
        left && `${left} left`, j.file_count > 1 && `file ${j.file_index} of ${j.file_count}`].filter(Boolean).join(" · ");
    } else if (j.status === "done") line = `installed · ${bytesText(j.total)}`;
    else if (j.status === "cancelled") line = "cancelled · what arrived is kept, and downloading again resumes";
    else line = j.error ?? "failed";

    const head = el("div", "md-job-head");
    head.append(el("span", "md-job-name", j.model ?? `${j.repo}${j.quant ? `:${j.quant}` : ""}`),
      el("span", "md-job-pct", running && j.total ? `${Math.floor(share * 100)}%` : ""));
    const track = el("div", `md-track${j.status === "resolving" ? " indeterminate" : ""}`);
    const fill = el("span", "md-fill");
    fill.style.width = `${(j.status === "done" ? 1 : share) * 100}%`;
    track.append(fill);
    const foot = el("div", "md-job-foot");
    const acts = el("span", "md-job-acts");
    foot.append(el("span", "md-job-line", line), acts);
    row.append(head, track, foot);

    const button = (label, fn) => {
      const b = el("button", "linky", label);
      b.type = "button";
      b.addEventListener("click", fn);
      acts.append(b);
    };
    if (running) {
      button("Cancel", async () => { await api(`/api/hub/pulls/${j.id}`, { method: "DELETE" }); pollJobs(); });
    } else {
      if (j.status === "failed" || j.status === "cancelled") {
        button(j.status === "failed" ? "Retry" : "Resume", async () => {
          await api(`/api/hub/pulls/${j.id}`, { method: "DELETE" });
          startPull(j.repo, j.quant, j.alias);
        });
      }
      button("Dismiss", async () => { await api(`/api/hub/pulls/${j.id}`, { method: "DELETE" }); pollJobs(); });
    }
    box.append(row);
  }
}

function renderInstalled() {
  const box = $("md-installed");
  box.replaceChildren();
  if (!hub.installed.length) {
    box.append(el("p", "hint", "Nothing installed yet. Search above, or install a file you have."));
    return;
  }
  for (const m of hub.installed) {
    const row = el("div", "md-model");
    const main = el("div", "md-model-main");
    const chips = el("span", "md-chips");
    if (m.is_default) chips.append(el("span", "md-chip md-good", "default"));
    if (m.vision) chips.append(el("span", "md-chip", "vision"));
    if (m.embedding) chips.append(el("span", "md-chip md-faint", "embedding"));
    main.append(el("span", "md-model-name", m.alias || m.reference), chips);
    const meta = el("div", "md-model-meta",
      [m.alias ? m.reference : null, bytesText(m.size_bytes), m.context_train ? `${countText(m.context_train)} context` : null].filter(Boolean).join(" · "));
    const acts = el("div", "md-model-acts");
    if (!m.embedding && !m.is_default) {
      const d = el("button", "ghost-btn auto", "Make default");
      d.type = "button";
      d.addEventListener("click", async () => {
        d.disabled = true;
        try {
          await api("/api/admin/models/default", { method: "POST", body: JSON.stringify({ model: m.alias || m.reference }) });
          hub.installed = await api("/api/models");
          renderInstalled();
        } catch (x) { d.textContent = x.message; }
      });
      acts.append(d);
    }
    const del = el("button", "ghost-btn auto md-danger", "Delete");
    del.type = "button";
    del.addEventListener("click", () => confirmDelete(row, m));
    acts.append(del);
    row.append(main, meta, acts);
    box.append(row);
  }
}

function confirmDelete(row, m) {
  const acts = row.querySelector(".md-model-acts");
  acts.replaceChildren();
  const ask = el("span", "md-ask", `Delete ${bytesText(m.size_bytes)} from disk?`);
  const yes = el("button", "ghost-btn auto md-danger", "Delete");
  const no = el("button", "ghost-btn auto", "Keep");
  yes.type = no.type = "button";
  no.addEventListener("click", renderInstalled);
  yes.addEventListener("click", async () => {
    yes.disabled = true;
    try {
      await api(`/api/models/${encodeURIComponent(m.reference)}`, { method: "DELETE" });
      hub.installed = await api("/api/models");
      renderInstalled();
      if (hub.repo) renderQuants();
    } catch (e) {
      ask.textContent = e.message;
      ask.classList.add("error");
    }
  });
  acts.append(ask, no, yes);
}

$("md-query").addEventListener("input", () => {
  clearTimeout(hub.searchTimer);
  hub.searchTimer = setTimeout(() => searchHub($("md-query").value), 350);
});
$("md-search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  clearTimeout(hub.searchTimer);
  const direct = asRepo($("md-query").value);
  if (direct) openRepo(direct.split(":")[0]);
  else searchHub($("md-query").value);
});
$("md-repo-back").addEventListener("click", () => {
  $("md-repo").hidden = true;
  $("md-results").hidden = false;
});
$("md-pull").addEventListener("click", () => {
  if (hub.repo && hub.chosen) startPull(hub.repo.id, hub.chosen, $("md-alias").value.trim() || null);
});

// ------------------------------------------------------- install from a file

const up = { xhr: null };

/// The first four bytes of every GGUF file. Checked here so a wrong file is
/// refused before gigabytes of it are sent.
async function isGguf(file) {
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return String.fromCharCode(...head) === "GGUF";
}

for (const [input, label, empty] of [["up-weights", "up-weights-name", "Choose the model file…"], ["up-mmproj", "up-mmproj-name", "Projector (optional)"]]) {
  $(input).addEventListener("change", async () => {
    const f = $(input).files[0];
    $(label).textContent = f ? `${f.name} · ${bytesText(f.size)}` : empty;
    $("up-error").hidden = true;
    if (f && !(await isGguf(f))) {
      $("up-error").textContent = `${f.name} is not a GGUF file. Safetensors and PyTorch checkpoints must be converted first.`;
      $("up-error").hidden = false;
      $(input).value = "";
      $(label).textContent = empty;
    }
    $("up-go").disabled = !$("up-weights").files[0];
  });
}

/// Send one file, reporting progress. Resolves to the server's id for it.
function sendFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    up.xhr = xhr;
    xhr.open("PUT", `/api/admin/models/upload?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("x-ozgent-admin", "1");
    xhr.setRequestHeader("content-type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded, e.total); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { body = null; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.error ?? `${xhr.status} ${xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("the upload was interrupted"));
    xhr.onabort = () => reject(new Error("cancelled"));
    xhr.send(file);
  });
}

$("up-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const weights = $("up-weights").files[0];
  const mmproj = $("up-mmproj").files[0];
  if (!weights) return;
  const err = $("up-error");
  err.hidden = true;
  const box = $("up-progress");
  box.hidden = false;
  box.className = "md-job downloading";
  $("up-go").disabled = true;
  $("up-cancel").hidden = false;

  const total = weights.size + (mmproj?.size ?? 0);
  const started = performance.now();
  let before = 0;
  const show = (sent, what) => {
    const secs = (performance.now() - started) / 1000;
    const rate = secs > 0.5 ? sent / secs : 0;
    const share = total ? sent / total : 0;
    $("up-what").textContent = what;
    $("up-pct").textContent = `${Math.floor(share * 100)}%`;
    $("up-fill").style.width = `${share * 100}%`;
    $("up-line").textContent = [`${bytesText(sent)} of ${bytesText(total)}`, rate ? `${bytesText(rate)}/s` : "starting",
      rate ? `${durationText((total - sent) / rate)} left` : ""].filter(Boolean).join(" · ");
  };
  try {
    const w = await sendFile(weights, (done) => show(done, weights.name));
    before = weights.size;
    let m = null;
    if (mmproj) m = await sendFile(mmproj, (done) => show(before + done, mmproj.name));
    // The bytes are here; registering is quick but not instant.
    $("up-cancel").hidden = true;
    $("up-line").textContent = "installing…";
    $("up-pct").textContent = "";
    box.querySelector(".md-track").classList.add("indeterminate");
    const out = await api("/api/admin/models/import", {
      method: "POST",
      body: JSON.stringify({ weights: w.id, mmproj: m?.id ?? null, alias: $("up-alias").value.trim() || null }),
    });
    box.querySelector(".md-track").classList.remove("indeterminate");
    box.className = "md-job done";
    $("up-fill").style.width = "100%";
    $("up-line").textContent = `installed as ${out.model}${out.vision ? " · vision" : ""}`;
    $("up-weights").value = "";
    $("up-mmproj").value = "";
    $("up-alias").value = "";
    $("up-weights-name").textContent = "Choose the model file…";
    $("up-mmproj-name").textContent = "Projector (optional)";
    hub.installed = await api("/api/models");
    renderInstalled();
  } catch (x) {
    box.querySelector(".md-track").classList.remove("indeterminate");
    box.className = "md-job failed";
    $("up-line").textContent = x.message === "cancelled" ? "cancelled" : "failed";
    if (x.message !== "cancelled") {
      err.textContent = x.message;
      err.hidden = false;
    }
  } finally {
    up.xhr = null;
    $("up-cancel").hidden = true;
    $("up-go").disabled = !$("up-weights").files[0];
  }
});

$("up-cancel").addEventListener("click", () => up.xhr?.abort());

// ================================================================= password

$("pw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const note = $("pw-note");
  note.className = "hint inline";
  if ($("pw-new").value !== $("pw-again").value) {
    note.textContent = "The two new passwords are different.";
    note.classList.add("error");
    return;
  }
  note.textContent = "Saving…";
  try {
    await api("/api/admin/password", {
      method: "POST",
      body: JSON.stringify({ current: $("pw-current").value, new: $("pw-new").value }),
    });
    for (const id of ["pw-current", "pw-new", "pw-again"]) $(id).value = "";
    note.textContent = "Changed. Other browsers are signed out.";
  } catch (x) {
    note.textContent = x.message;
    note.classList.add("error");
  }
});

boot();

// ================================================================== security

const lines = (id) => $(id).value.split(/\n/).map((s) => s.trim()).filter(Boolean);
const words = (id) => $(id).value.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);

async function openSecurity() {
  try {
    const [net, sb, srv] = await Promise.all([api("/api/admin/access"), api("/api/admin/sandbox"), api("/api/admin/server")]);
    renderNet(net);
    renderKeys(net);
    renderSandbox(sb);
    $("sb-timeout").value = srv.tool_timeout_seconds;
    $("sb-calls").value = srv.max_calls_per_turn;
    $("sb-handoff").checked = srv.handoff;
  } catch (e) {
    $("net-note").textContent = e.message;
  }
}

function renderNet(n) {
  $("net-you").textContent = n.you ?? "unknown";
  for (const r of document.querySelectorAll('input[name="net-mode"]')) r.checked = r.value === n.mode;
  $("net-allow").value = n.allow.join("\n");
  $("net-deny").value = n.deny.join("\n");
  $("net-hosts").value = n.hosts.join("\n");
  $("net-proxies").value = n.trusted_proxies.join("\n");
  $("net-rpm").value = n.requests_per_minute;
  $("net-conns").value = n.max_connections_per_address;
  $("net-fails").value = n.max_auth_failures;
  $("net-lock").value = n.lockout_minutes;
  $("net-body").value = n.max_body_mb;
  $("net-local-api").checked = n.local_api_open;
}

async function saveNet(force = false) {
  const note = $("net-note");
  const body = {
    mode: document.querySelector('input[name="net-mode"]:checked')?.value ?? "open",
    allow: lines("net-allow"),
    deny: lines("net-deny"),
    hosts: lines("net-hosts"),
    trusted_proxies: lines("net-proxies"),
    requests_per_minute: Number($("net-rpm").value || 0),
    max_connections_per_address: Number($("net-conns").value || 0),
    max_auth_failures: Number($("net-fails").value || 0),
    lockout_minutes: Number($("net-lock").value || 1),
    max_body_mb: Number($("net-body").value || 1),
    local_api_open: $("net-local-api").checked,
    force,
  };
  note.textContent = "saving…";
  try {
    await api("/api/admin/access", { method: "PUT", body: JSON.stringify(body) });
    note.textContent = "saved";
  } catch (e) {
    if (/refuse your own address/.test(e.message) && confirm(`${e.message}\n\nSave anyway?`)) return saveNet(true);
    note.textContent = e.message;
  }
}

$("net-form").addEventListener("submit", (e) => { e.preventDefault(); saveNet(); });

function renderKeys(n) {
  const list = $("key-list");
  list.replaceChildren();
  if (!n.keys.length) list.append(el("p", "hint", "No keys yet."));
  for (const k of n.keys) {
    const row = el("div", `sec-key${k.disabled ? " off" : ""}`);
    const head = el("div", "sec-key-head");
    head.append(el("b", null, k.name), el("code", null, `${k.id}…`));
    if (k.disabled) head.append(el("span", "adm-pill", "disabled"));
    const scopes = el("div", "hint", k.scopes.map((s) => n.scopes.find((d) => d.name === s)?.describes ?? s).join(" · "));
    const made = el("div", "hint", k.created ? `created ${new Date(k.created * 1000).toLocaleString()}` : "");
    const actions = el("div", "adm-actions");
    const toggle = el("button", "ghost-btn auto", k.disabled ? "Enable" : "Disable");
    toggle.type = "button";
    toggle.addEventListener("click", async () => {
      await api(`/api/admin/keys/${encodeURIComponent(k.id)}`, { method: "PATCH", body: JSON.stringify({ disabled: !k.disabled }) });
      openSecurity();
    });
    const revoke = el("button", "ghost-btn auto danger", "Revoke");
    revoke.type = "button";
    revoke.addEventListener("click", async () => {
      if (!confirm(`Revoke "${k.name}"? Anything using it stops working.`)) return;
      await api(`/api/admin/keys/${encodeURIComponent(k.id)}`, { method: "DELETE" });
      openSecurity();
    });
    actions.append(toggle, revoke);
    row.append(head, scopes, made, actions);
    list.append(row);
  }
  const box = $("key-scopes");
  if (!box.querySelector("input")) {
    for (const s of n.scopes) {
      const label = el("label", "check");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = s.name;
      input.checked = s.name === "inference";
      label.append(input, document.createTextNode(` ${s.describes}`));
      box.append(label);
    }
  }
}

$("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const scopes = [...$("key-scopes").querySelectorAll("input:checked")].map((i) => i.value);
  const note = $("key-note");
  try {
    const made = await api("/api/admin/keys", { method: "POST", body: JSON.stringify({ name: $("key-name").value, scopes }) });
    $("key-value").textContent = made.key;
    $("key-reveal").hidden = false;
    $("key-name").value = "";
    note.textContent = "";
    openSecurity();
  } catch (err) {
    note.textContent = err.message;
  }
});

$("key-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("key-value").textContent); $("key-copy").textContent = "Copied"; } catch (_) { /* select it by hand */ }
});
$("key-done").addEventListener("click", () => {
  $("key-value").textContent = "";
  $("key-reveal").hidden = true;
  $("key-copy").textContent = "Copy";
});

function renderSandbox(sb) {
  $("sb-root").value = sb.root ?? "";
  $("sb-write").checked = sb.write;
  $("sb-shell").checked = sb.shell;
  $("sb-allow").value = sb.shell_allow.join(" ");
  $("sb-shell-net").checked = sb.shell_network;
  $("sb-net").checked = sb.network;
  $("sb-net-allow").value = sb.network_allow.join(" ");
  $("sb-private").checked = sb.network_private;
  $("sb-sensitive").checked = sb.allow_sensitive;
  const mcp = sb.mcp_servers.length ? sb.mcp_servers.join(", ") : "none";
  const extra = sb.extra_paths.length ? sb.extra_paths.join(", ") : "none";
  $("sb-fixed").textContent =
    `Interpreter: ${sb.python}. Extra tool folders: ${extra}. MCP servers: ${mcp}. ` +
    "Each names a program that runs with your rights, so they change only in config.toml.";
}

$("sb-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const note = $("sb-note");
  const root = $("sb-root").value.trim();
  note.textContent = "saving…";
  try {
    await api("/api/admin/sandbox", {
      method: "PUT",
      body: JSON.stringify({
        root: root || null,
        write: $("sb-write").checked,
        shell: $("sb-shell").checked,
        shell_allow: words("sb-allow"),
        shell_network: $("sb-shell-net").checked,
        network: $("sb-net").checked,
        network_allow: words("sb-net-allow"),
        network_private: $("sb-private").checked,
        allow_sensitive: $("sb-sensitive").checked,
      }),
    });
    await api("/api/admin/server", {
      method: "PUT",
      body: JSON.stringify({
        tool_timeout_seconds: Number($("sb-timeout").value || 30),
        max_calls_per_turn: Number($("sb-calls").value || 8),
        handoff: $("sb-handoff").checked,
      }),
    });
    note.textContent = "saved; tools restarted";
  } catch (err) {
    note.textContent = err.message;
  }
});

// ================================================================== embeddings

async function openEmbedding() {
  try {
    const e = await api("/api/admin/embedding");
    $("emb-enabled").checked = e.enabled;
    const sel = $("emb-model");
    sel.replaceChildren();
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = e.installed.length ? `Automatic (${e.installed[0]})` : "Automatic — none installed";
    sel.append(auto);
    for (const m of e.installed) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      sel.append(o);
    }
    sel.value = e.model ?? "";
    $("emb-device").value = e.device;
    $("emb-max").value = e.max_tokens;
    const s = e.status;
    let text;
    if (!e.enabled) text = "Off: memory recalls by keywords.";
    else if (s.error) text = `Not working: ${s.error}`;
    else if (s.model) text = `${s.model}, ${s.dimensions} dimensions, on the ${s.device.toUpperCase()}.`;
    else if (e.chosen) text = `${e.chosen} — loads the first time memory needs it.`;
    else text = "No embedding model installed: memory recalls by keywords. Get one above — for example Qwen/Qwen3-Embedding-0.6B-GGUF.";
    if (e.coverage) text += ` ${e.coverage.done} of ${e.coverage.all} messages have vectors.`;
    if (e.backfilling) text += " Embedding earlier messages now…";
    $("emb-status").textContent = text;
  } catch (err) {
    $("emb-status").textContent = err.message;
  }
}

$("emb-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const note = $("emb-note");
  note.textContent = "saving…";
  try {
    await api("/api/admin/embedding", {
      method: "PUT",
      body: JSON.stringify({
        enabled: $("emb-enabled").checked,
        model: $("emb-model").value || null,
        device: $("emb-device").value,
        max_tokens: Number($("emb-max").value || 0),
      }),
    });
    note.textContent = "saved; it reloads on next use";
    openEmbedding();
  } catch (err) {
    note.textContent = err.message;
  }
});

$("emb-backfill").addEventListener("click", async () => {
  await api("/api/admin/embedding/backfill", { method: "POST" });
  $("emb-note").textContent = "embedding earlier messages in the background";
  setTimeout(openEmbedding, 1500);
});

// ================================================================== server

async function openServer() {
  try {
    const s = await api("/api/admin/server");
    $("srv-idle").value = s.idle_unload_minutes;
    $("srv-parallel").value = s.parallel;
  } catch (err) {
    $("srv-note").textContent = err.message;
  }
}

$("srv-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const note = $("srv-note");
  try {
    await api("/api/admin/server", {
      method: "PUT",
      body: JSON.stringify({
        idle_unload_minutes: Number($("srv-idle").value || 0),
        parallel: Number($("srv-parallel").value || 1),
      }),
    });
    note.textContent = "saved";
  } catch (err) {
    note.textContent = err.message;
  }
});

// ================================================================== mcp
//
// MCP servers: programs and services that give the model more tools. The
// page never says what to run: a registry install names the entry and the
// server looks it up again; every add is reviewed (the exact command, from
// the server) before it can be saved. Secret values never come back here.

const mcp = {
  data: null,
  /// Cards with unsaved edits are not re-rendered under the cursor.
  dirty: new Set(),
  way: "registry",
  manualKind: "command",
  picked: null,
  next: null,
};

const MCP_STATE = {
  connected: ["connected", "good"],
  failed: ["failed", "bad"],
  disabled: ["switched off", ""],
  off: ["off", ""],
};

const RULE_TEXT = { allow: "runs without asking", ask: "asks first", deny: "never runs" };

async function loadMcp() {
  clearTimeout(timers.mcp);
  try {
    mcp.data = await api("/api/admin/mcp");
    renderMcp();
  } catch (e) {
    if (e.message !== "signed out") mcpDetail(e.message, "bad");
  }
  const d = mcp.data;
  const waiting = d && (d.reconnecting ||
    (d.enabled && d.tools_enabled && d.servers.some((s) => s.enabled && !s.status && !s.invalid)));
  if (waiting && !$("view-mcp").hidden) timers.mcp = setTimeout(loadMcp, 1500);
}

function mcpDetail(text, tone) {
  const box = $("mcp-detail");
  box.hidden = !text;
  box.textContent = text ?? "";
  box.className = `adm-detail ${tone ?? ""}`;
}

function renderMcp() {
  const d = mcp.data;
  $("mcp-enabled").checked = d.enabled;
  const pill = $("mcp-pill");
  const connected = d.servers.filter((s) => s.status?.state === "connected").length;
  const failed = d.servers.filter((s) => s.status?.state === "failed" || s.invalid).length;
  if (!d.tools_enabled) { pill.textContent = "tools off"; pill.className = "adm-pill"; }
  else if (!d.enabled) { pill.textContent = "off"; pill.className = "adm-pill"; }
  else if (d.reconnecting) { pill.textContent = "connecting"; pill.className = "adm-pill live"; }
  else if (failed) { pill.textContent = `${connected} connected · ${failed} failed`; pill.className = "adm-pill bad"; }
  else { pill.textContent = `${connected} connected`; pill.className = `adm-pill ${connected ? "good" : ""}`; }

  mcpDetail(!d.tools_enabled
    ? "Tools are switched off in the chat page's Settings, so no server runs until they are on."
    : "", "");
  const r = d.runtimes;
  const have = ["npx", "uvx", "docker"].filter((k) => r[k]);
  const lack = ["npx", "uvx", "docker"].filter((k) => !r[k]);
  $("mcp-runtimes").textContent =
    (have.length ? `Can run ${have.join(", ")} packages.` : "Neither npx nor uvx is installed.") +
    (lack.length ? ` Not installed: ${lack.join(", ")}.` : "") +
    (r.sandbox ? "" : " The sandbox is unavailable: ozgent's Python runtime was not found.");

  const box = $("mcp-servers");
  const existing = new Map([...box.children].filter((c) => c.dataset.name).map((c) => [c.dataset.name, c]));
  const cards = [];
  for (const server of d.servers) {
    const old = existing.get(server.name);
    if (old && mcp.dirty.has(server.name)) {
      setServerPill(old.querySelector("[data-role=pill]"), server, d);
      cards.push(old);
    } else {
      cards.push(serverCard(server, d));
    }
  }
  if (!cards.length) {
    const empty = el("div", "adm-card mcp-empty",
      "No servers yet. Add one from the MCP Registry, as an npm or PyPI package, or as a command or URL.");
    cards.push(empty);
  }
  box.replaceChildren(...cards);
}

function setServerPill(pill, s, d) {
  let label, tone = "";
  if (s.invalid) { label = "misconfigured"; tone = "bad"; }
  else if (!d.tools_enabled || !d.enabled) { label = "off"; }
  else if (!s.enabled) { label = "switched off"; }
  else if (!s.status) { label = "connecting"; tone = "live"; }
  else {
    [label, tone] = MCP_STATE[s.status.state] ?? [s.status.state, ""];
    if (s.status.state === "connected") {
      const offered = s.tools.filter((t) => t.offered).length;
      label = `connected · ${offered} tool${offered === 1 ? "" : "s"}`;
    }
  }
  pill.textContent = label;
  pill.className = `adm-pill ${tone}`;
}

function row(label, ...body) {
  const r = el("div", "adm-row");
  r.append(el("div", "adm-row-label", label));
  const b = el("div", "adm-row-body");
  b.append(...body);
  r.append(b);
  return r;
}

function toggle(checked, text) {
  const label = el("label", "adm-check");
  const sw = el("span", "switch");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  sw.append(input, el("span", "track"));
  const t = el("span");
  t.innerHTML = text;
  label.append(sw, t);
  return [label, input];
}

function serverCard(s, d) {
  const card = el("article", "adm-card adm-channel mcp-card");
  card.dataset.name = s.name;
  const markDirty = () => { mcp.dirty.add(s.name); save.disabled = false; note.textContent = ""; };

  // ---- head
  const head = el("header", "adm-ch-head");
  head.append(el("h2", null, s.name));
  const pill = el("span", "adm-pill");
  pill.dataset.role = "pill";
  setServerPill(pill, s, d);
  head.append(pill);
  if (s.sandbox) head.append(el("span", "adm-pill sandboxed", "sandboxed"));
  head.append(el("span", "spacer"));
  const sw = el("label", "switch");
  sw.dataset.tip = s.enabled ? "Switch this server off" : "Switch this server on";
  const on = document.createElement("input");
  on.type = "checkbox";
  on.checked = s.enabled;
  on.setAttribute("aria-label", `Use ${s.name}`);
  on.addEventListener("change", async () => {
    try { await api(`/api/admin/mcp/servers/${encodeURIComponent(s.name)}`, { method: "PATCH", body: JSON.stringify({ enabled: on.checked }) }); }
    catch (e) { on.checked = !on.checked; mcpDetail(e.message, "bad"); }
    loadMcp();
  });
  sw.append(on, el("span", "track"));
  head.append(sw);
  card.append(head);

  if (s.description) card.append(el("p", "adm-detail", s.description));
  if (s.used === false) {
    card.append(el("p", "adm-detail",
      "Running, but left out of the chats: it is unticked in the chat page's Settings → Tools."));
  }
  const problem = s.invalid ?? s.status?.error;
  if (problem) {
    const p = el("p", "adm-detail bad", problem);
    card.append(p);
    if (s.sandbox && s.status?.state === "failed") {
      card.append(el("p", "adm-detail",
        "It runs in the sandbox, which keeps your home folder, other programs' temporary files and your session " +
        "out of its reach. If it needs a folder of yours, add it under Sandbox below. To check whether the sandbox " +
        "is the cause at all, switch it off, save, and see if it connects."));
    }
  }

  // ---- what runs
  const run = el("div", "mcp-run", s.transport === "http" ? s.url : [s.command, ...s.args].join(" "));
  const runBody = [run];
  if (s.source) runBody.push(el("span", "mcp-source", s.source));
  if (s.status?.server) runBody.push(el("span", "mcp-source", `reports itself as ${s.status.server}`));
  if (s.status?.log?.length) {
    const log = el("details", "mcp-log");
    log.append(el("summary", null, `Its last ${s.status.log.length} lines of output`));
    log.append(el("pre", null, s.status.log.join("\n")));
    if (s.status.state === "failed") log.open = true;
    runBody.push(log);
  }
  card.append(row(s.transport === "http" ? "Connects to" : "Runs", ...runBody));

  // ---- tools
  const toolsBox = el("div", "mcp-tools");
  const toolInputs = [];
  for (const t of s.tools) {
    const r = el("div", `mcp-tool${t.offered ? "" : " off"}`);
    const pick = document.createElement("input");
    pick.type = "checkbox";
    pick.checked = t.offered;
    pick.setAttribute("aria-label", `Offer ${t.name}`);
    const name = el("span", "mcp-tool-name", t.name);
    name.append(el("span", "mcp-tool-effect", t.effect));
    const rule = document.createElement("select");
    rule.className = "select";
    rule.setAttribute("aria-label", `When the model calls ${t.name}`);
    const def = document.createElement("option");
    def.value = "";
    def.textContent = t.own_rule ? "default rule"
      : `${s.server_rule ? "server's rule" : "default"}: ${RULE_TEXT[t.rule] ?? t.rule}`;
    rule.append(def);
    for (const [v, text] of Object.entries(RULE_TEXT)) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = text;
      rule.append(o);
    }
    rule.value = t.own_rule ?? "";
    pick.addEventListener("change", () => { r.classList.toggle("off", !pick.checked); markDirty(); });
    rule.addEventListener("change", markDirty);
    r.append(pick, name, rule);
    if (t.description) {
      const d = el("span", "mcp-tool-desc", t.description);
      d.title = t.description;
      r.append(d);
    }
    toolsBox.append(r);
    toolInputs.push({ t, pick, rule });
  }
  const toolHint = el("span", "hint inline", s.tools.length
    ? "Untick a tool to keep it from the model. A tool that asks first cannot be used by API keys or scheduled jobs, since nobody is there to answer; allow it to let them."
    : s.status?.state === "connected" ? "It offers no tools." : "Its tools are listed once it connects.");
  if (s.tools.length) {
    // Folded: a server can list dozens, and the page is a list of servers.
    const offered = s.tools.filter((t) => t.offered).length;
    const allowed = s.tools.filter((t) => t.offered && t.rule === "allow").length;
    const fold = el("details", "mcp-tools-fold");
    fold.append(el("summary", null,
      `${s.tools.length} tool${s.tools.length === 1 ? "" : "s"}` +
      (offered === s.tools.length ? ", all offered" : `, ${offered} offered`) +
      (allowed ? ` · ${allowed} run without asking` : " · every one asks first")));
    fold.append(toolsBox, toolHint);
    fold.open = mcp.openTools?.has(s.name) ?? false;
    fold.addEventListener("toggle", () => {
      mcp.openTools ??= new Set();
      if (fold.open) mcp.openTools.add(s.name); else mcp.openTools.delete(s.name);
    });
    card.append(row("Tools", fold));
  } else {
    card.append(row("Tools", toolHint));
  }

  // ---- one rule for all of them: fifty tools are one decision, not fifty.
  // A rule set on a single tool above still wins over this.
  const serverRule = document.createElement("select");
  serverRule.className = "select";
  serverRule.setAttribute("aria-label", `When the model calls any of ${s.name}'s tools`);
  for (const [v, text] of [["", "each tool by its own rule"], ...Object.entries(RULE_TEXT)]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v ? `every tool ${text}` : text;
    serverRule.append(o);
  }
  serverRule.value = s.server_rule ?? "";
  serverRule.addEventListener("change", markDirty);
  card.append(row("All its tools", serverRule, el("span", "hint inline",
    "Without a rule here, a tool without its own rule asks first unless the server is trusted. " +
    "A rule set on a single tool in the list above still wins.")));

  // ---- sandbox
  let sandbox, network, folders;
  if (s.transport !== "http") {
    const [sbLabel, sbInput] = toggle(s.sandbox,
      "<b>Run it in the sandbox.</b> It gets a home of its own and can't read your files, credentials or other programs.");
    const [netLabel, netInput] = toggle(s.network, "It may use the network (needed to download the package).");
    const fold = el("textarea", "text-input st-mono mcp-lines");
    fold.rows = 2;
    fold.placeholder = "Folders it may use, one per line";
    fold.value = (s.folders ?? []).join("\n");
    for (const i of [sbInput, netInput, fold]) i.addEventListener("input", markDirty);
    sandbox = sbInput; network = netInput; folders = fold;
    card.append(row("Sandbox", sbLabel, netLabel, fold));
  }

  // ---- environment or headers: names only; values are replaced, never read.
  const pairName = s.transport === "http" ? "headers" : "env";
  const changes = {};
  const pairs = el("div", "mcp-pairs");
  for (const key of s[pairName]) {
    const p = el("div", "mcp-pair");
    const value = el("input", "text-input");
    value.type = "password";
    value.placeholder = "set · type to replace";
    value.autocomplete = "off";
    value.addEventListener("input", () => { changes[key] = value.value; markDirty(); });
    const x = el("button", "adm-chip-x", "×");
    x.type = "button";
    x.setAttribute("aria-label", `Remove ${key}`);
    x.addEventListener("click", () => { changes[key] = null; p.remove(); markDirty(); });
    p.append(el("span", "mcp-pair-name", key), value, x);
    pairs.append(p);
  }
  const addPair = el("div", "mcp-pair");
  const newName = el("input", "text-input st-mono");
  newName.placeholder = pairName === "env" ? "NEW_VARIABLE" : "Header-Name";
  newName.spellcheck = false;
  const newValue = el("input", "text-input");
  newValue.type = "password";
  newValue.placeholder = "value";
  newValue.autocomplete = "off";
  for (const i of [newName, newValue]) i.addEventListener("input", markDirty);
  addPair.append(newName, newValue, el("span"));
  pairs.append(addPair);
  card.append(row(pairName === "env" ? "Environment" : "Headers", pairs));

  // ---- trust and time
  const [trustLabel, trust] = toggle(s.trust_hints,
    "Believe what it says about its own tools: one it calls read-only then runs under your <i>read</i> rule. Only for a server you run yourself.");
  trust.addEventListener("input", markDirty);
  const timeout = el("input", "text-input");
  timeout.type = "number";
  timeout.min = "1";
  timeout.max = "3600";
  timeout.value = s.timeout_seconds;
  timeout.style.maxWidth = "120px";
  timeout.addEventListener("input", markDirty);
  card.append(row("Trust", trustLabel));
  card.append(row("Seconds per call", timeout));
  const load = el("select", "select");
  for (const [v, text] of [["auto", "Automatic"], ["always", "Always"], ["on_request", "Looked up when needed"]]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = text;
    load.append(o);
  }
  load.value = s.load ?? "auto";
  load.style.maxWidth = "260px";
  load.addEventListener("change", markDirty);
  card.append(row("Tools up front", load, el("span", "hint inline",
    "Every tool described up front takes context before anyone has said anything, and a long list makes a model " +
    "worse at choosing. Automatic describes this server's tools up front while all the tools fit a budget; past " +
    "it, the model sees their names, and each message brings the few it is most likely about in full. Its tools " +
    "can be called either way.")));

  // ---- actions
  const actions = el("div", "adm-actions");
  const save = el("button", "primary-btn", "Save");
  save.type = "button";
  save.disabled = true;
  const note = el("span", "hint inline");
  const remove = el("button", "ghost-btn auto", "Remove");
  remove.type = "button";
  let armed = null;
  remove.addEventListener("click", async () => {
    if (!armed) {
      remove.textContent = `Remove ${s.name} and its tools?`;
      remove.classList.add("st-delete");
      armed = setTimeout(() => { armed = null; remove.textContent = "Remove"; remove.classList.remove("st-delete"); }, 4000);
      return;
    }
    clearTimeout(armed);
    try {
      await api(`/api/admin/mcp/servers/${encodeURIComponent(s.name)}`, { method: "DELETE" });
      mcp.dirty.delete(s.name);
    } catch (e) { note.textContent = e.message; note.className = "error"; }
    loadMcp();
  });
  save.addEventListener("click", async () => {
    const body = {};
    if (toolInputs.length) {
      const offered = toolInputs.filter((x) => x.pick.checked).map((x) => x.t.remote);
      const all = offered.length === toolInputs.length;
      const before = s.only;
      const same = all ? before == null : before && before.length === offered.length && offered.every((o) => before.includes(o));
      if (!same) body.only = all ? null : offered;
      const rules = {};
      for (const { t, rule } of toolInputs) if ((t.own_rule ?? "") !== rule.value) rules[t.name] = rule.value;
      if (Object.keys(rules).length) body.rules = rules;
    }
    if ((s.server_rule ?? "") !== serverRule.value) {
      body.rules = { ...(body.rules ?? {}), [`${s.name}_*`]: serverRule.value };
    }
    if (sandbox) {
      body.sandbox = sandbox.checked;
      body.network = network.checked;
      body.folders = folders.value.split("\n").map((f) => f.trim()).filter(Boolean);
    }
    body.trust_hints = trust.checked;
    body.timeout_seconds = Number(timeout.value) || s.timeout_seconds;
    body.load = load.value;
    const pairChanges = { ...changes };
    if (newName.value.trim()) pairChanges[newName.value.trim()] = newValue.value;
    if (Object.keys(pairChanges).length) body[pairName] = pairChanges;
    save.disabled = true;
    try {
      await api(`/api/admin/mcp/servers/${encodeURIComponent(s.name)}`, { method: "PATCH", body: JSON.stringify(body) });
      mcp.dirty.delete(s.name);
      note.className = "hint inline";
      note.textContent = "Saved.";
      loadMcp();
    } catch (e) {
      save.disabled = false;
      note.className = "error";
      note.textContent = e.message;
    }
  });
  actions.append(save, note, el("span", "spacer"), remove);
  card.append(actions);
  return card;
}

$("mcp-enabled").addEventListener("change", async (e) => {
  try { await api("/api/admin/mcp", { method: "PUT", body: JSON.stringify({ enabled: e.target.checked }) }); }
  catch (err) { e.target.checked = !e.target.checked; mcpDetail(err.message, "bad"); }
  loadMcp();
});

$("mcp-reconnect").addEventListener("click", async () => {
  try { await api("/api/admin/mcp/reconnect", { method: "POST" }); } catch (e) { mcpDetail(e.message, "bad"); }
  setTimeout(loadMcp, 300);
});

// ------------------------------------------------------------ add a server

function mcpSetWay(way) {
  mcp.way = way;
  for (const b of $("mcp-way").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.way === way));
  for (const p of $("mcp-dialog").querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== way;
  mcpInvalidate();
  mcpSandboxVisibility();
  $("mcp-common").hidden = way === "registry" && !mcp.picked;
  // Pasted JSON names its servers; the field is only for a lone one.
  $("mcp-name").placeholder = way === "json" ? "only for a single unnamed server" : "files";
}

function mcpSetManualKind(kind) {
  mcp.manualKind = kind;
  for (const b of $("mcp-manual-kind").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.kind === kind));
  for (const p of $("mcp-dialog").querySelectorAll("[data-kind-panel]")) p.hidden = p.dataset.kindPanel !== kind;
  mcpInvalidate();
  mcpSandboxVisibility();
}

/// A URL has no program to sandbox.
function mcpSandboxVisibility() {
  let remote = false;
  if (mcp.way === "manual") remote = mcp.manualKind === "url";
  if (mcp.way === "registry" && mcp.picked) remote = mcp.picked.options[Number($("mcp-option").value)]?.kind === "remote";
  $("mcp-sandbox-box").hidden = remote;
}

/// Anything edited after a review needs reviewing again before it saves.
function mcpInvalidate() {
  $("mcp-review").hidden = true;
  $("mcp-save").disabled = true;
  $("mcp-save").textContent = "Add server";
  $("mcp-error").hidden = true;
}

function mcpNameExample() {
  const n = $("mcp-name").value.trim() || "files";
  $("mcp-name-eg").textContent = `${n}_read_file`;
}

function openMcpDialog() {
  mcp.picked = null;
  for (const id of ["mcp-search", "mcp-name", "mcp-json", "mcp-pkg-name", "mcp-pkg-version", "mcp-pkg-args", "mcp-pkg-env",
                    "mcp-cmd", "mcp-cmd-args", "mcp-cmd-env", "mcp-url", "mcp-url-headers", "mcp-folders"]) $(id).value = "";
  $("mcp-sandbox").checked = true;
  $("mcp-network").checked = true;
  $("mcp-results").replaceChildren();
  $("mcp-more").hidden = true;
  $("mcp-pick").hidden = true;
  mcpSetManualKind("command");
  mcpSetWay("registry");
  mcpNameExample();
  $("mcp-dialog").showModal();
  $("mcp-search").focus();
}

async function mcpSearch(more) {
  const q = $("mcp-search").value.trim();
  const box = $("mcp-results");
  $("mcp-pick").hidden = true;
  box.hidden = false;
  mcp.picked = null;
  $("mcp-common").hidden = true;
  if (!more) {
    box.replaceChildren(el("div", "mcp-empty", "Searching the registry… its search can take up to half a minute."));
  }
  try {
    const cursor = more && mcp.next ? `&cursor=${encodeURIComponent(mcp.next)}` : "";
    const res = await api(`/api/admin/mcp/registry?search=${encodeURIComponent(q)}${cursor}`);
    if (!more) box.replaceChildren();
    for (const l of res.servers) box.append(mcpResult(l));
    if (!box.children.length) box.append(el("div", "mcp-empty", "Nothing in the registry matches that."));
    mcp.next = res.next;
    $("mcp-more").hidden = !res.next || !res.servers.length;
  } catch (e) {
    box.replaceChildren(el("div", "mcp-empty", e.message));
  }
}

function mcpResult(l) {
  const b = el("button", "mcp-result");
  b.type = "button";
  b.append(el("span", "mcp-result-name", l.title || l.name.split("/").pop()));
  const kinds = el("span", "mcp-kinds");
  for (const o of l.options) {
    const k = el("span", `mcp-kind${o.supported && o.runner_present ? "" : " no"}`, o.kind);
    k.title = !o.supported ? o.why_not : !o.runner_present ? `needs ${o.runner}, which is not installed` : o.identifier;
    kinds.append(k);
  }
  b.append(kinds);
  b.append(el("span", "mcp-result-id", `${l.name} · ${l.version}`));
  if (l.description) b.append(el("span", "mcp-result-desc", l.description));
  b.addEventListener("click", () => mcpPick(l));
  return b;
}

function mcpPick(l) {
  mcp.picked = l;
  $("mcp-results").hidden = true;
  $("mcp-more").hidden = true;
  $("mcp-pick").hidden = false;
  $("mcp-common").hidden = false;
  const head = $("mcp-pick-head");
  head.replaceChildren(el("h3", null, l.title || l.name));
  head.append(el("p", null, `${l.name} · version ${l.version}`));
  if (l.description) head.append(el("p", null, l.description));
  const src = l.repository || l.website;
  if (src) {
    const a = el("a", null, `Source: ${src}`);
    a.href = src;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    head.append(a);
  } else {
    head.append(el("p", "error", "It lists no source repository."));
  }
  const select = $("mcp-option");
  select.replaceChildren();
  let first = -1;
  l.options.forEach((o, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    const where = o.kind === "remote" ? o.identifier : `${o.kind} · ${o.identifier}${o.version ? `@${o.version}` : ""}`;
    opt.textContent = o.supported && o.runner_present ? where : `${where} (can't: ${o.why_not ?? `needs ${o.runner}`})`;
    opt.disabled = !(o.supported && o.runner_present);
    if (!opt.disabled && first < 0) first = i;
    select.append(opt);
  });
  select.value = String(Math.max(first, 0));
  $("mcp-option-why").hidden = first >= 0;
  $("mcp-option-why").textContent = "None of its ways to run can be used here.";
  $("mcp-name").value = l.suggested_name;
  mcpNameExample();
  mcpInputs();
}

function mcpInputs() {
  const o = mcp.picked?.options[Number($("mcp-option").value)];
  const box = $("mcp-inputs");
  box.replaceChildren();
  for (const input of o?.inputs ?? []) {
    const f = el("label", "st-field");
    const title = el("span", null, input.name);
    if (!input.required) title.append(el("em", "st-optional", "optional"));
    f.append(title);
    let control;
    if (input.choices?.length) {
      control = el("select", "select");
      for (const c of input.choices) { const opt = document.createElement("option"); opt.value = c; opt.textContent = c; control.append(opt); }
      if (input.default) control.value = input.default;
    } else {
      control = el("input", "text-input st-mono");
      control.type = input.secret ? "password" : "text";
      control.autocomplete = "off";
      control.spellcheck = false;
      if (input.default && !input.secret) control.value = input.default;
    }
    control.dataset.key = input.key;
    control.addEventListener("input", mcpInvalidate);
    f.append(control);
    if (input.description) f.append(el("small", null, input.description));
    box.append(f);
  }
  mcpSandboxVisibility();
  mcpInvalidate();
}

function mcpLines(id) {
  return $(id).value.split("\n").map((x) => x.trim()).filter(Boolean);
}

function mcpPairs(id, sep) {
  const out = {};
  for (const line of mcpLines(id)) {
    const at = line.indexOf(sep);
    if (at < 1) throw new Error(`"${line}" needs a ${sep === "=" ? "NAME=value" : "Name: value"} form`);
    out[line.slice(0, at).trim()] = line.slice(at + 1).trim();
  }
  return out;
}

function mcpRequest(preview) {
  const common = {
    name: $("mcp-name").value.trim(),
    sandbox: $("mcp-sandbox").checked,
    network: $("mcp-network").checked,
    folders: mcpLines("mcp-folders"),
    preview,
  };
  if (mcp.way === "json") {
    if (!$("mcp-json").value.trim()) throw new Error("Paste the JSON first.");
    return ["/api/admin/mcp/import", { ...common, name: common.name || null, text: $("mcp-json").value }];
  }
  if (mcp.way === "registry") {
    if (!mcp.picked) throw new Error("Choose a server from the results first.");
    const values = {};
    for (const c of $("mcp-inputs").querySelectorAll("[data-key]")) if (c.value.trim()) values[c.dataset.key] = c.value;
    return ["/api/admin/mcp/install", {
      ...common,
      registry_name: mcp.picked.name,
      version: mcp.picked.version,
      option: Number($("mcp-option").value),
      values,
    }];
  }
  if (mcp.way === "package") {
    return ["/api/admin/mcp/servers", {
      ...common,
      kind: $("mcp-pkg-kind").value,
      package: $("mcp-pkg-name").value.trim(),
      version: $("mcp-pkg-version").value.trim() || null,
      args: mcpLines("mcp-pkg-args"),
      env: mcpPairs("mcp-pkg-env", "="),
    }];
  }
  if (mcp.manualKind === "url") {
    return ["/api/admin/mcp/servers", {
      ...common,
      kind: "url",
      url: $("mcp-url").value.trim(),
      headers: mcpPairs("mcp-url-headers", ":"),
    }];
  }
  return ["/api/admin/mcp/servers", {
    ...common,
    kind: "command",
    command: $("mcp-cmd").value.trim(),
    args: mcpLines("mcp-cmd-args"),
    env: mcpPairs("mcp-cmd-env", "="),
  }];
}

async function mcpSubmit(preview) {
  const error = $("mcp-error");
  error.hidden = true;
  try {
    if (!$("mcp-name").value.trim() && mcp.way !== "json") throw new Error("Give it a name.");
    const [path, body] = mcpRequest(preview);
    const res = await api(path, { method: "POST", body: JSON.stringify(body) });
    if (res.servers) {
      // Pasted JSON: one line per server, and the ones that cannot come over.
      const lines = res.servers.map((s) => s.error
        ? `✗ ${s.name}: ${s.error}`
        : `${s.name}:  ${s.preview}${s.sandbox ? "  (in the sandbox)" : ""}` +
          (s.folders?.length ? `\n    may use ${s.folders.join(", ")}` : "") +
          (s.notes?.length ? `\n    note: ${s.notes.join("; ")}` : ""));
      const usable = res.servers.filter((s) => !s.error).length;
      if (preview) {
        $("mcp-review-cmd").textContent = lines.join("\n");
        $("mcp-review").hidden = false;
        $("mcp-save").disabled = !usable;
        $("mcp-save").textContent = usable > 1 ? `Add ${usable} servers` : "Add server";
      } else {
        $("mcp-dialog").close();
        loadMcp();
      }
      return;
    }
    if (preview) {
      $("mcp-review-cmd").textContent = res.preview + (res.sandbox ? "\n(in the sandbox)" : "");
      $("mcp-review").hidden = false;
      $("mcp-save").disabled = false;
    } else {
      $("mcp-dialog").close();
      loadMcp();
    }
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  }
}

for (const b of $("mcp-way").querySelectorAll("button")) b.addEventListener("click", () => mcpSetWay(b.dataset.way));
for (const b of $("mcp-manual-kind").querySelectorAll("button")) b.addEventListener("click", () => mcpSetManualKind(b.dataset.kind));
$("mcp-add").addEventListener("click", openMcpDialog);
$("mcp-cancel").addEventListener("click", () => $("mcp-dialog").close());
$("mcp-check").addEventListener("click", () => mcpSubmit(true));
$("mcp-save").addEventListener("click", () => mcpSubmit(false));
$("mcp-search-go").addEventListener("click", () => mcpSearch(false));
$("mcp-more").addEventListener("click", () => mcpSearch(true));
$("mcp-search").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); mcpSearch(false); } });
$("mcp-back").addEventListener("click", () => {
  mcp.picked = null;
  $("mcp-pick").hidden = true;
  $("mcp-results").hidden = false;
  $("mcp-more").hidden = !mcp.next;
  $("mcp-common").hidden = true;
  mcpInvalidate();
});
$("mcp-option").addEventListener("change", mcpInputs);
$("mcp-name").addEventListener("input", () => { mcpNameExample(); mcpInvalidate(); });
$("mcp-pkg-name").addEventListener("blur", () => {
  if ($("mcp-name").value.trim()) return;
  const last = $("mcp-pkg-name").value.trim().split("/").pop() ?? "";
  $("mcp-name").value = last.replace(/^(mcp-server-|server-)/, "").replace(/(-mcp-server|-mcp)$/, "")
    .replace(/[^A-Za-z0-9_-]/g, "-").slice(0, 24);
  mcpNameExample();
});
$("mcp-save").textContent = "Add server";
for (const id of ["mcp-json", "mcp-pkg-kind", "mcp-pkg-name", "mcp-pkg-version", "mcp-pkg-args", "mcp-pkg-env", "mcp-cmd", "mcp-cmd-args",
                  "mcp-cmd-env", "mcp-url", "mcp-url-headers", "mcp-folders", "mcp-sandbox", "mcp-network"]) {
  $(id).addEventListener("input", mcpInvalidate);
}
