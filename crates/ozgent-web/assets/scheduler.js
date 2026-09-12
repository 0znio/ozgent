// ozgent /scheduler — jobs, when they run, and how they went.
//
// The page does more than the `schedule` tool a chat uses, on purpose: a chat
// is for the sentence you are in the middle of, this is for reviewing fifteen
// jobs and finding the one that has been quietly failing since Tuesday.

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers ?? {}) },
  });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (!res.ok) throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
  return body;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

/// A unix time as a short local string. The server sends instants; only the
/// browser knows what the reader's clock says.
function when(unix) {
  if (!unix) return "";
  const d = new Date(unix * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return sameDay ? time : `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

let state = { jobs: [], agents: [], channels: [], allowed: {}, zone: "local", hosted: false };
let editing = null;   // the job being changed, or null when creating

// ------------------------------------------------------------------- list

async function load() {
  try {
    state = await api("/api/scheduler");
  } catch (e) {
    banner(`Could not read the scheduler: ${e.message}`, "bad");
    return;
  }
  fillAgents();
  fillBanner();
  draw();
}

function banner(text, kind) {
  const b = $("banner");
  b.hidden = !text;
  b.className = `adm-banner${kind ? ` ${kind}` : ""}`;
  b.textContent = text ?? "";
}

function fillBanner() {
  if (state.hosted) return banner(null);
  // Perfectly good jobs and nothing awake to run them is the failure people
  // stare at without diagnosing, so it is said first and plainly.
  if (state.elsewhere) {
    return banner(`${state.elsewhere} is running these jobs, not this one. They will still fire.`, "");
  }
  banner(
    "Nothing is running these jobs. Start ozgent in the background with " +
      "`ozgent daemon install`, or leave `ozgent web` open.",
    "bad"
  );
}

function fillAgents() {
  const sel = $("f-agent");
  const chosen = sel.value;
  sel.replaceChildren();
  sel.append(new Option("the default model", ""));
  for (const a of state.agents ?? []) sel.append(new Option(`@${a}`, a));
  sel.value = chosen;
}

function draw() {
  const list = $("list");
  list.replaceChildren();
  $("empty").hidden = state.jobs.length > 0;
  for (const job of state.jobs) list.append(card(job));
}

function card(job) {
  const c = el("article", `adm-card sch-job${job.enabled ? "" : " off"}`);

  const head = el("header", "sch-job-head");
  head.append(el("h2", null, job.name));
  head.append(statusPill(job));
  head.append(el("span", "spacer"));

  const sw = el("label", "switch");
  sw.dataset.tip = job.enabled ? "Pause this job" : "Start it again";
  const box = el("input");
  box.type = "checkbox";
  box.checked = job.enabled;
  box.setAttribute("aria-label", `Run ${job.name}`);
  box.onchange = () => change(job.name, { enabled: box.checked });
  sw.append(box, el("span", "track"));
  head.append(sw);
  c.append(head);

  c.append(el("p", "sch-when", job.when_words));
  c.append(el("p", "sch-asks", job.prompt));

  const chips = el("div", "adm-chips sch-chips");
  if (job.agent) chips.append(el("span", "adm-chip-item", `@${job.agent}`));
  chips.append(
    el(
      "span",
      "adm-chip-item",
      job.deliver === "none"
        ? "kept on this page"
        : `→ ${job.deliver}${job.deliver_to ? ` · ${job.deliver_to}` : " · everyone allowed"}`
    )
  );
  if (job.only_if) chips.append(el("span", "adm-chip-item", `only if: ${job.only_if}`));
  if (job.zone && job.zone !== "local") chips.append(el("span", "adm-chip-item", job.zone));
  c.append(chips);

  const foot = el("div", "sch-job-foot");
  const next = el("span", "sch-next");
  if (!job.enabled) next.textContent = "paused";
  else if (job.next_run_at) next.textContent = `next ${job.next_in} · ${when(job.next_run_at)}`;
  else next.textContent = "no next run";
  foot.append(next);
  foot.append(el("span", "spacer"));

  foot.append(button("Runs", "i-clock", () => showRuns(job.name), job.runs === 0));
  foot.append(button("Run now", "i-play", () => runNow(job.name)));
  foot.append(button("Edit", "i-edit", () => openForm(job)));
  foot.append(button("Delete", "i-trash", () => remove(job)));
  c.append(foot);
  return c;
}

function button(label, icon, onclick, disabled) {
  const b = el("button", "ghost-btn auto sch-act");
  b.type = "button";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ic");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${icon}`);
  svg.append(use);
  b.append(svg, document.createTextNode(label));
  b.disabled = !!disabled;
  b.onclick = onclick;
  return b;
}

/// How the last run went, as one word. A job failing since Tuesday has to be
/// visible without opening it.
function statusPill(job) {
  const map = {
    ok: ["ok", "good"],
    quiet: ["quiet", ""],
    error: ["failing", "bad"],
    missed: ["missed", "warn"],
    running: ["running", "live"],
  };
  const [text, kind] = map[job.last_status] ?? ["never run", ""];
  const p = el("span", `adm-pill${kind ? ` ${kind}` : ""}`, text);
  if (job.failures > 1) p.textContent = `failing ×${job.failures}`;
  return p;
}

// ---------------------------------------------------------------- actions

async function change(name, body) {
  try {
    await api(`/api/scheduler/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
  } catch (e) {
    banner(e.message, "bad");
  }
  load();
}

async function runNow(name) {
  try {
    const out = await api(`/api/scheduler/${encodeURIComponent(name)}/run`, { method: "POST" });
    // Never claim it ran: it is queued, and with nothing hosting it is queued
    // indefinitely.
    banner(
      out.hosted
        ? `${name} will run within a minute. Its answer goes where the job says.`
        : `${name} is due, but nothing is running jobs yet.`,
      out.hosted ? "" : "bad"
    );
  } catch (e) {
    banner(e.message, "bad");
  }
  load();
}

async function remove(job) {
  if (!confirm(`Delete ${job.name} and its history?`)) return;
  try {
    await api(`/api/scheduler/${encodeURIComponent(job.name)}`, { method: "DELETE" });
  } catch (e) {
    banner(e.message, "bad");
  }
  load();
}

// ------------------------------------------------------------------- form

function openForm(job) {
  editing = job ?? null;
  $("form-title").textContent = job ? `Edit ${job.name}` : "New job";
  $("save").textContent = job ? "Save" : "Schedule it";
  $("f-name").value = job?.name ?? "";
  $("f-prompt").value = job?.prompt ?? "";
  $("f-when").value = job?.when ?? "";
  $("f-agent").value = job?.agent ?? "";
  $("f-deliver").value = job?.deliver ?? "none";
  $("f-to").value = job?.deliver_to ?? "";
  $("f-only-if").value = job?.only_if ?? "";
  $("f-zone").value = job?.zone && job.zone !== "local" ? job.zone : "";
  $("zone-hint").textContent = `Local time here is ${state.zone}. An IANA name like Asia/Kolkata for anywhere else.`;
  $("form-error").hidden = true;
  $("preview").hidden = true;
  deliverChanged();
  $("sheet").hidden = false;
  $("f-name").focus();
  if (job) previewWhen();
}

function closeForm() {
  $("sheet").hidden = true;
  editing = null;
}

function deliverChanged() {
  const to = $("f-deliver").value;
  $("to-field").hidden = to === "none";
  if (to === "none") return;
  // Left empty it goes to everyone that channel allows, which is what someone
  // who already listed who may use it means. Nobody knows their own Telegram
  // chat id, so demanding one here was the wrong question.
  const live = (state.channels ?? []).includes(to);
  const who = (state.allowed ?? {})[to] ?? [];
  const audience = who.length
    ? `Leave it empty and the answer goes to everyone ${to} allows: ${who.join(", ")}.`
    : `Leave it empty and the answer goes to everyone ${to} allows.`;
  $("to-hint").textContent = live
    ? audience
    : `${audience} ${to} is not connected right now, so nothing arrives until it is.`;
}

/// Read the rule back from the server and show when it would actually fire.
let previewTimer = null;
function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(previewWhen, 350);
}

async function previewWhen() {
  const text = $("f-when").value.trim();
  const box = $("preview");
  if (!text) {
    box.hidden = true;
    return;
  }
  try {
    const out = await api("/api/scheduler/preview", {
      method: "POST",
      body: JSON.stringify({ when: text, zone: $("f-zone").value.trim() || null }),
    });
    box.className = "sch-preview";
    box.replaceChildren();
    box.append(el("div", "sch-preview-words", out.words));
    const times = el("ul", "sch-preview-times");
    for (const f of out.fires) {
      const li = el("li");
      li.append(el("span", "sch-preview-at", when(f.at)));
      li.append(el("span", "hint", f.in));
      times.append(li);
    }
    box.append(times);
    box.hidden = false;
  } catch (e) {
    box.className = "sch-preview bad";
    box.replaceChildren(el("div", null, e.message));
    box.hidden = false;
  }
}

async function submit(event) {
  event.preventDefault();
  const body = {
    name: $("f-name").value.trim(),
    prompt: $("f-prompt").value.trim(),
    when: $("f-when").value.trim(),
    agent: $("f-agent").value,
    only_if: $("f-only-if").value.trim(),
    deliver: $("f-deliver").value,
    deliver_to: $("f-to").value.trim(),
    zone: $("f-zone").value.trim() || "local",
  };
  const err = $("form-error");
  err.hidden = true;
  $("save").disabled = true;
  try {
    if (editing) {
      await api(`/api/scheduler/${encodeURIComponent(editing.name)}`, {
        method: "PUT",
        body: JSON.stringify(body),
      });
    } else {
      await api("/api/scheduler", { method: "POST", body: JSON.stringify(body) });
    }
    closeForm();
    load();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    $("save").disabled = false;
  }
}

// ------------------------------------------------------------------- runs

async function showRuns(name) {
  $("runs-title").textContent = name;
  const body = $("runs-body");
  body.replaceChildren(el("p", "hint", "loading…"));
  $("runs-sheet").hidden = false;
  let out;
  try {
    out = await api(`/api/scheduler/${encodeURIComponent(name)}`);
  } catch (e) {
    body.replaceChildren(el("p", "error", e.message));
    return;
  }
  body.replaceChildren();
  if (!out.runs.length) {
    body.append(el("p", "hint", "It has not run yet."));
    return;
  }
  if (out.job.conversation) {
    const link = el("a", "linky", "open the conversation these were written to");
    link.href = `/chat?c=${encodeURIComponent(out.job.conversation)}`;
    body.append(link);
  }
  for (const run of out.runs) {
    const r = el("div", `sch-run ${run.status}`);
    const head = el("div", "sch-run-head");
    head.append(el("span", "sch-run-when", when(run.started_at)));
    head.append(el("span", "adm-pill", runLabel(run)));
    r.append(head);
    if (run.error) r.append(el("p", "error", run.error));
    if (run.output) {
      const pre = el("pre", "sch-run-out", run.output);
      r.append(pre);
    }
    body.append(r);
  }
}

function runLabel(run) {
  if (run.status === "ok") return run.delivered ? "sent" : "answered";
  if (run.status === "quiet") return "nothing to say";
  if (run.status === "missed") return "ozgent was not running";
  if (run.status === "running") return "running now";
  return "failed";
}

// ------------------------------------------------------------------- wire

$("new-job").onclick = () => openForm(null);
$("empty-new").onclick = () => openForm(null);
$("form").onsubmit = submit;
$("f-when").oninput = schedulePreview;
$("f-zone").oninput = schedulePreview;
$("f-deliver").onchange = deliverChanged;
for (const node of document.querySelectorAll("[data-close]")) node.onclick = closeForm;
for (const node of document.querySelectorAll("[data-close-runs]")) {
  node.onclick = () => { $("runs-sheet").hidden = true; };
}
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("runs-sheet").hidden) $("runs-sheet").hidden = true;
  else if (!$("sheet").hidden) closeForm();
});

load();
// Jobs fire while the page is open, and a next-run time goes stale on its own.
setInterval(() => {
  if ($("sheet").hidden && $("runs-sheet").hidden) load();
}, 20000);
