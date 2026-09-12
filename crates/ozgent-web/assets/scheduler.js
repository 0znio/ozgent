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

// A unix time as a short local string. The server sends instants; only the
// browser knows what the reader's clock says.
function when_(unix) {
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
  else if (job.next_run_at) next.textContent = `next ${job.next_in} · ${when_(job.next_run_at)}`;
  else next.textContent = "no next run";
  foot.append(next);
  foot.append(el("span", "spacer"));

  foot.append(button("Run now", "i-play", () => runNow(job.name)));
  foot.append(button("Edit", "i-edit", () => openForm(job)));
  foot.append(button("Delete", "i-trash", () => remove(job)));
  c.append(foot);

  // The history opens in place rather than over the page. "Did it run, and
  // did it work" is the question people come here with, and answering it
  // should not cover up the job being asked about.
  const details = el("details", "sch-history");
  details.open = open.has(job.name);
  const summary = el("summary");
  summary.append(el("span", "sch-history-label", historyLabel(job)));
  details.append(summary);
  const body = el("div", "sch-history-body");
  details.append(body);
  details.addEventListener("toggle", () => {
    // Remembered so a reload every twenty seconds does not close a panel
    // somebody is reading.
    if (details.open) { open.add(job.name); fillHistory(body, job.name); }
    else open.delete(job.name);
  });
  if (details.open) fillHistory(body, job.name);
  c.append(details);
  return c;
}

// Which job histories are open, so redrawing does not shut them.
const open = new Set();

// What the closed row says, so it is worth opening — or worth not opening.
function historyLabel(job) {
  if (!job.runs) return "Never run";
  const when = job.last_run_at ? when_(job.last_run_at) : "";
  const outcome = {
    ok: job.deliver === "none" ? "answered" : "sent",
    quiet: "nothing to say",
    error: "failed",
    missed: "missed — ozgent was not running",
    running: "running now",
  }[job.last_status] ?? "ran";
  const times = job.runs === 1 ? "once" : `${job.runs} times`;
  return `Ran ${times} · last ${when} · ${outcome}`;
}

async function fillHistory(body, name) {
  body.replaceChildren(el("p", "hint", "loading…"));
  let out;
  try {
    out = await api(`/api/scheduler/${encodeURIComponent(name)}`);
  } catch (e) {
    body.replaceChildren(el("p", "error", e.message));
    return;
  }
  body.replaceChildren();

  // Where it goes, said here rather than only as a chip, because this is the
  // panel somebody opens when it did not arrive.
  const facts = el("dl", "sch-facts");
  const fact = (k, v) => { facts.append(el("dt", null, k), el("dd", null, v)); };
  fact("Delivery", out.job.deliver === "none"
    ? "kept on this page"
    : `${out.job.deliver} · ${out.job.deliver_to || "everyone allowed"}`);
  fact("Asked of", out.job.agent ? `@${out.job.agent}` : "the default model");
  fact("Next run", out.job.enabled
    ? (out.job.next_run_at ? `${out.job.next_in} · ${when_(out.job.next_run_at)}` : "none")
    : "paused");
  if (out.job.only_if) fact("Only when", out.job.only_if);
  body.append(facts);

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
    head.append(el("span", "sch-run-when", when_(run.started_at)));
    head.append(el("span", `adm-pill ${runKind(run)}`, runLabel(run)));
    if (run.finished_at) {
      head.append(el("span", "hint", `${run.finished_at - run.started_at}s`));
    }
    r.append(head);
    if (run.error) r.append(el("p", "error", run.error));
    if (run.output) r.append(el("pre", "sch-run-out", run.output));
    body.append(r);
  }
}

// The pill colour for a run, so a failure is visible while scrolling.
function runKind(run) {
  return { ok: "good", error: "bad", missed: "warn", running: "live" }[run.status] ?? "";
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

// How the last run went, as one word. A job failing since Tuesday has to be
// visible without opening it.
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
  let hosted = false;
  try {
    const out = await api(`/api/scheduler/${encodeURIComponent(name)}/run`, { method: "POST" });
    hosted = out.hosted;
    // Never claim it ran: it is queued, and with nothing hosting it is queued
    // indefinitely.
    banner(
      hosted
        ? `${name} is starting. Its answer goes where the job says.`
        : `${name} is due, but nothing is running jobs yet.`,
      hosted ? "" : "bad"
    );
  } catch (e) {
    banner(e.message, "bad");
    return;
  }
  await load();
  if (hosted) chase(name);
}

// Watch one job closely for a few seconds.
//
// The list refreshes every twenty seconds, which is right for jobs firing on
// their own and far too slow just after a button was pressed: the run starts
// at once and the page still said nothing about it for most of a minute,
// which reads as a button that did not work. This follows it until it is
// running, then hands back to the ordinary refresh — the run itself can take
// minutes, and polling for all of it would be a page that never rests.
async function chase(name) {
  for (const wait of [500, 1000, 1500, 2000, 3000, 4000]) {
    await new Promise((r) => setTimeout(r, wait));
    if (!$("sheet").hidden) return;
    await load();
    const job = (state.jobs ?? []).find((j) => j.name === name);
    if (job?.last_status === "running") return;
  }
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

// Read the rule back from the server and show when it would actually fire.
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
      li.append(el("span", "sch-preview-at", when_(f.at)));
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
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("sheet").hidden) closeForm();
});

load();
// Jobs fire while the page is open, and a next-run time goes stale on its own.
setInterval(() => {
  if ($("sheet").hidden) load();
}, 20000);
