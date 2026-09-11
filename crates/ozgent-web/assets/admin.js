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

let timers = { gateway: null, jobs: null };
function stopPolling() {
  clearTimeout(timers.gateway);
  clearTimeout(timers.jobs);
}

function openView(name) {
  if (!["gateway", "models", "account"].includes(name)) name = "gateway";
  for (const t of document.querySelectorAll(".adm-nav .tab")) {
    t.setAttribute("aria-selected", String(t.dataset.view === name));
  }
  for (const v of document.querySelectorAll(".adm-view")) v.hidden = v.id !== `view-${name}`;
  history.replaceState(null, "", `#${name}`);
  stopPolling();
  if (name === "gateway") pollGateway();
  if (name === "models") openModels();
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
