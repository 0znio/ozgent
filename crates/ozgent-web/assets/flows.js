/* Workflows — the canvas.
 *
 * A graph editor is mostly bookkeeping about coordinates, and the interesting
 * decisions are the few places where it is not:
 *
 * - The model is one plain object, `flow`, and everything on screen is drawn
 *   from it. Dragging a step writes to the model and redraws; nothing is ever
 *   read back out of the DOM. That is what makes save, undo-by-reload and the
 *   run overlay all work without a second source of truth.
 * - Steps are HTML and wires are SVG, sharing one transformed coordinate
 *   space. HTML because a step is a small form with text in it, and laying that
 *   out in SVG is a fight; SVG because a bezier between two moving points is
 *   one path attribute.
 * - The palette is fetched, not written here. Every installed tool is a step,
 *   with a form built from the schema it already declares, so adding a tool to
 *   `~/ozgent/tools` adds a node to this editor and nothing here changes.
 */

(() => {
  const SPACE = 8000; // half the wire canvas, which is centred on the origin

  /** The kinds this editor can place, and what to say about them. */
  const KINDS = {
    manual:    { group: "Starts with",  label: "When I press run",  what: "You start it from here." },
    schedule:  { group: "Starts with",  label: "On a schedule",     what: "Every so often, or daily." },
    webhook:   { group: "Starts with",  label: "When called",       what: "Something POSTs to a URL." },
    agent:     { group: "Then",         label: "Ask the model",     what: "A prompt; the reply is the output." },
    tool:      { group: "Then",         label: "Use a tool",        what: "Any tool you have installed." },
    condition: { group: "Then",         label: "Only if",           what: "Sends the run one of two ways." },
    text:      { group: "Then",         label: "Build some text",   what: "Stitch earlier outputs together." },
  };

  const OPS = [
    ["eq", "is"], ["ne", "is not"], ["contains", "contains"],
    ["gt", "is more than"], ["lt", "is less than"],
    ["empty", "is empty"], ["not_empty", "is not empty"],
  ];

  const view = {
    flow: null,          // the graph being edited
    id: null,            // its row id
    uuid: null,
    enabled: false,
    palette: { tools: [], models: [] },
    selected: null,      // step id
    pan: { x: 0, y: 0 },
    zoom: 1,
    run: null,           // step id -> { status, ms, error }
    dirty: false,
    saving: false,
  };

  const h = (tag, attrs = {}, ...kids) => {
    const node = tag === "svg" || tag === "path"
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k === "class") node.setAttribute("class", v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function call(path, options = {}) {
    const res = await fetch(path, {
      headers: options.body ? { "content-type": "application/json" } : {},
      ...options,
    });
    if (!res.ok) {
      let message = res.statusText;
      try { message = (await res.json()).error || message; } catch {}
      throw new Error(message);
    }
    return res.status === 204 ? null : res.json();
  }

  // ------------------------------------------------------------ the list

  async function showList(host) {
    const flows = await call("/api/flows");
    host.replaceChildren(
      h("div", { class: "fl-flows-view" },
        h("div", { class: "fl-flow-head" },
          h("strong", {}, "Workflows"),
          h("span", { class: "fl-spacer" }),
          h("button", { class: "primary-btn", onclick: makeFlow }, "New workflow")),
        h("div", { class: "fl-flow-list" },
          h("h1", {}, "Workflows"),
          h("p", { class: "fl-lede" },
            "Steps wired together: ask the model, use a tool, branch on the answer. " +
            "Run one by hand, on a schedule, or when something calls its URL."),
          flows.length === 0
            ? h("p", { class: "fl-lede" }, "Nothing here yet.")
            : flows.map((f) => h("button", {
                class: "fl-flow-card",
                onclick: () => go(`/flows/${f.id}`),
              },
              h("span", {},
                h("span", { class: "fl-name" }, f.name),
                h("span", { class: "fl-meta" },
                  h("span", {}, `${f.steps} step${f.steps === 1 ? "" : "s"}`),
                  f.valid ? null : h("span", { class: "fl-broken" }, "not finished"))),
              f.enabled ? h("span", { class: "fl-live" }, "live") : h("span", {}))))));
  }

  async function makeFlow() {
    const made = await call("/api/flows", { method: "POST", body: JSON.stringify({}) });
    go(`/flows/${made.id}`);
  }

  // ---------------------------------------------------------- the editor

  async function showEditor(host, id) {
    const [detail, palette] = await Promise.all([
      call(`/api/flows/${id}`),
      call("/api/flows/palette").catch(() => ({ tools: [], models: [] })),
    ]);
    view.flow = detail.definition;
    view.id = detail.id;
    view.uuid = detail.uuid;
    view.enabled = detail.enabled;
    view.palette = palette;
    view.selected = null;
    view.run = null;
    view.dirty = false;
    view.pan = { x: 0, y: 0 };
    view.zoom = 1;

    host.replaceChildren(
      h("div", { class: "fl-flows-view" },
        header(),
        h("div", { class: "fl-body", id: "flow-body" },
          h("div", { class: "fl-canvas", id: "canvas" },
            h("div", { class: "fl-canvas-space", id: "space" },
              svgLayer())),
          h("aside", { class: "fl-inspector", id: "inspector" }))));

    wireCanvas();
    draw();
    loadRuns();
  }

  function header() {
    return h("div", { class: "fl-flow-head" },
      h("button", { class: "ghost-btn auto", onclick: () => go("/flows") }, "← All"),
      h("input", {
        class: "fl-flow-name", id: "flow-name", value: view.flow.name || "",
        oninput: (e) => { view.flow.name = e.target.value; touch(); },
      }),
      h("span", { class: "fl-problems", id: "flow-problems" }),
      h("span", { class: "fl-spacer" }),
      h("span", { class: "fl-saved", id: "flow-saved" }),
      h("button", { class: "ghost-btn auto", id: "add-step" }, "+ Step"),
      h("label", { class: "ghost-btn auto fl-live-toggle", title: "Let schedules and webhooks fire" },
        h("input", {
          type: "checkbox", id: "flow-enabled", ...(view.enabled ? { checked: true } : {}),
          onchange: (e) => { view.enabled = e.target.checked; save(); },
        }),
        h("span", {}, "Live")),
      h("button", { class: "ghost-btn auto", onclick: save }, "Save"),
      h("button", { class: "primary-btn", id: "run-flow", onclick: runFlow }, "Run"));
  }

  function svgLayer() {
    // Centred on the origin and huge, so panning never runs off the drawing
    // surface — simpler and steadier than resizing it as the graph grows.
    return h("svg", { class: "fl-wires", id: "wires", viewBox: `${-SPACE} ${-SPACE} ${SPACE * 2} ${SPACE * 2}` });
  }

  // ------------------------------------------------------------- drawing

  function draw() {
    const space = document.getElementById("space");
    if (!space) return;
    space.style.transform = `translate(${view.pan.x}px, ${view.pan.y}px) scale(${view.zoom})`;

    const canvas = document.getElementById("canvas");
    // The grid moves and scales with the graph, or panning looks like the
    // steps are sliding over a stationary floor.
    canvas.style.backgroundSize = `${24 * view.zoom}px ${24 * view.zoom}px`;
    canvas.style.backgroundPosition = `${view.pan.x}px ${view.pan.y}px`;

    for (const old of space.querySelectorAll(".fl-step")) old.remove();
    for (const node of view.flow.nodes) space.append(stepEl(node));
    drawWires();
    showProblems();
    showInspector();
  }

  function stepEl(node) {
    const outcome = view.run?.[node.id];
    const kind = node.kind;
    const classes = ["fl-step"];
    if (view.selected === node.id) classes.push("fl-selected");
    if (outcome) classes.push(`fl-${outcome.status}`);

    const el = h("div", {
      class: classes.join(" "),
      "data-id": node.id,
      style: `left:${node.x}px; top:${node.y}px`,
    },
      h("div", { class: "fl-kind" },
        h("span", {}, kind),
        outcome?.ms != null ? h("span", { class: "fl-ms" }, `${outcome.ms}ms`) : null),
      h("div", { class: "fl-label" }, node.name || node.id),
      detailLine(node) ? h("div", { class: "fl-detail" }, detailLine(node)) : null,
      outcome?.error ? h("div", { class: "fl-why" }, outcome.error) : null);

    if (kind !== "manual" && kind !== "schedule" && kind !== "webhook") {
      el.append(h("div", { class: "fl-port fl-in", "data-port": "in" }));
    }
    if (kind === "condition") {
      el.append(
        h("div", { class: "fl-port fl-out fl-yes", "data-port": "yes" }),
        h("div", { class: "fl-port-tag fl-yes" }, "yes"),
        h("div", { class: "fl-port fl-out fl-no", "data-port": "no" }),
        h("div", { class: "fl-port-tag fl-no" }, "no"));
    } else {
      el.append(h("div", { class: "fl-port fl-out fl-single", "data-port": "out" }));
    }
    return el;
  }

  /** The one line of settings worth seeing without opening the step. */
  function detailLine(node) {
    switch (node.kind) {
      case "tool": return node.params?.tool || "fl-no tool chosen";
      case "agent": return clip(node.params?.prompt, 40) || "fl-no prompt";
      case "text": return clip(node.params?.template, 40) || "fl-no text";
      case "schedule": return node.params?.at ? `daily ${node.params.at} UTC` : (node.params?.every || "fl-no interval");
      case "webhook": return `/hooks/${view.uuid}/${node.id}`;
      case "condition": {
        const op = OPS.find(([k]) => k === (node.params?.op || "eq"));
        return `${clip(node.params?.left, 14)} ${op ? op[1] : ""} ${clip(node.params?.right, 14)}`.trim();
      }
      default: return "";
    }
  }

  const clip = (s, n) => {
    const text = String(s ?? "").replace(/\s+/g, " ").trim();
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  };

  /** Where a port sits, in canvas coordinates. */
  function portAt(nodeId, port) {
    const node = view.flow.nodes.find((n) => n.id === nodeId);
    if (!node) return null;
    const el = document.querySelector(`.fl-step[data-id="${CSS.escape(nodeId)}"]`);
    const width = el ? el.offsetWidth : 208;
    const height = el ? el.offsetHeight : 72;
    if (port === "in") return { x: node.x, y: node.y + height / 2 };
    if (port === "yes") return { x: node.x + width, y: node.y + height * 0.36 };
    if (port === "no") return { x: node.x + width, y: node.y + height * 0.68 };
    return { x: node.x + width, y: node.y + height / 2 };
  }

  /** A horizontal bezier, so wires leave and arrive flat. */
  function curve(a, b) {
    const reach = Math.max(40, Math.abs(b.x - a.x) * 0.5);
    return `M ${a.x} ${a.y} C ${a.x + reach} ${a.y}, ${b.x - reach} ${b.y}, ${b.x} ${b.y}`;
  }

  function drawWires(draft) {
    const svg = document.getElementById("wires");
    if (!svg) return;
    svg.replaceChildren();

    for (const [i, edge] of view.flow.edges.entries()) {
      const from = portAt(edge.from, edge.port);
      const to = portAt(edge.to, "in");
      if (!from || !to) continue;
      const d = curve(from, to);

      let state = "";
      if (view.run) {
        const source = view.run[edge.from];
        // A wire is live only if its source actually left by this port —
        // which is exactly the rule the engine runs on, so the picture and
        // the run agree by construction rather than by being kept in step.
        state = source?.status === "ran" && source.port === edge.port ? "fl-taken" : "fl-dead";
      }
      svg.append(
        h("path", { class: `fl-wire ${state}`, d }),
        h("path", {
          class: "fl-wire-hit", d,
          onclick: (e) => { e.stopPropagation(); view.flow.edges.splice(i, 1); touch(); draw(); },
        }));
    }
    if (draft) svg.append(h("path", { class: "fl-wire fl-draft", d: curve(draft.from, draft.to) }));
  }

  function showProblems() {
    const slot = document.getElementById("flow-problems");
    if (!slot) return;
    const problems = validate(view.flow);
    slot.textContent = problems.length ? problems[0] : "";
    slot.title = problems.join("\n");
  }

  /** The same rules the server enforces, so the canvas says so immediately. */
  function validate(flow) {
    const out = [];
    const ids = new Set(flow.nodes.map((n) => n.id));
    if (!flow.nodes.some((n) => ["manual", "schedule", "webhook"].includes(n.kind))) {
      out.push("nothing starts this flow: add a trigger");
    }
    for (const e of flow.edges) {
      if (!ids.has(e.from) || !ids.has(e.to)) out.push("a wire leads nowhere");
    }
    for (const n of flow.nodes) {
      if (n.kind === "tool" && !n.params?.tool) out.push(`${n.name || n.id}: no tool chosen`);
      if (n.kind === "agent" && !n.params?.prompt) out.push(`${n.name || n.id}: no prompt`);
      if (n.kind === "schedule" && !n.params?.every && !n.params?.at) {
        out.push(`${n.name || n.id}: no interval`);
      }
    }
    return out;
  }

  // -------------------------------------------------------- interactions

  function wireCanvas() {
    const canvas = document.getElementById("canvas");
    let drag = null;

    canvas.addEventListener("pointerdown", (e) => {
      const port = e.target.closest(".fl-port");
      const step = e.target.closest(".fl-step");

      if (port && step) {
        const which = port.dataset.port;
        if (which === "in") return; // wires are drawn from an output
        drag = { kind: "wire", from: step.dataset.id, port: which };
        canvas.classList.add("fl-wiring");
        port.classList.add("fl-armed");
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      if (step) {
        const node = view.flow.nodes.find((n) => n.id === step.dataset.id);
        select(node.id);
        drag = { kind: "step", node, dx: 0, dy: 0, from: pointIn(e) };
        canvas.setPointerCapture(e.pointerId);
        return;
      }

      select(null);
      drag = { kind: "pan", startX: e.clientX - view.pan.x, startY: e.clientY - view.pan.y };
      canvas.classList.add("fl-panning");
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      if (drag.kind === "pan") {
        view.pan = { x: e.clientX - drag.startX, y: e.clientY - drag.startY };
        draw();
        return;
      }
      if (drag.kind === "step") {
        const now = pointIn(e);
        drag.node.x += now.x - drag.from.x;
        drag.node.y += now.y - drag.from.y;
        drag.from = now;
        const el = document.querySelector(`.fl-step[data-id="${CSS.escape(drag.node.id)}"]`);
        if (el) { el.style.left = `${drag.node.x}px`; el.style.top = `${drag.node.y}px`; }
        drawWires();
        touch();
        return;
      }
      if (drag.kind === "wire") {
        drawWires({ from: portAt(drag.from, drag.port), to: pointIn(e) });
      }
    });

    const finish = (e) => {
      if (!drag) return;
      if (drag.kind === "wire") {
        const step = e.target.closest?.(".fl-step");
        if (step && step.dataset.id !== drag.from) connect(drag.from, drag.port, step.dataset.id);
      }
      canvas.classList.remove("fl-panning", "fl-wiring");
      canvas.querySelectorAll(".fl-port.fl-armed").forEach((p) => p.classList.remove("fl-armed"));
      drag = null;
      draw();
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const before = pointIn(e);
      view.zoom = Math.min(2, Math.max(0.35, view.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
      const after = pointIn(e);
      // Zoom about the cursor: keep whatever is under it under it.
      view.pan.x += (after.x - before.x) * view.zoom;
      view.pan.y += (after.y - before.y) * view.zoom;
      draw();
    }, { passive: false });

    document.getElementById("add-step").addEventListener("click", (e) => {
      e.stopPropagation();
      openAdder(e.currentTarget);
    });

    // Removed first: opening a second flow without this leaves two handlers
    // bound, and Delete then removes a step and immediately tries again.
    document.removeEventListener("keydown", onKey);
    document.addEventListener("keydown", onKey);
  }

  /** Pointer position in canvas coordinates. */
  function pointIn(e) {
    const box = document.getElementById("canvas").getBoundingClientRect();
    return {
      x: (e.clientX - box.left - view.pan.x) / view.zoom,
      y: (e.clientY - box.top - view.pan.y) / view.zoom,
    };
  }

  function connect(from, port, to) {
    const target = view.flow.nodes.find((n) => n.id === to);
    if (!target || ["manual", "schedule", "webhook"].includes(target.kind)) return;
    const already = view.flow.edges.some((e) => e.from === from && e.to === to && e.port === port);
    if (already) return;
    view.flow.edges.push({ from, to, port });
    touch();
  }

  function onKey(e) {
    if (!view.flow) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    if (typing) return;
    if ((e.key === "Delete" || e.key === "Backspace") && view.selected) {
      e.preventDefault();
      removeStep(view.selected);
    }
    if ((e.key === "s" || e.key === "S") && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      save();
    }
  }

  function removeStep(id) {
    view.flow.nodes = view.flow.nodes.filter((n) => n.id !== id);
    view.flow.edges = view.flow.edges.filter((e) => e.from !== id && e.to !== id);
    view.selected = null;
    touch();
    draw();
  }

  function select(id) {
    view.selected = id;
    document.getElementById("flow-body").classList.toggle("fl-inspecting", id != null);
    draw();
  }

  // ---------------------------------------------------------- adding steps

  function openAdder(anchor) {
    document.querySelector(".fl-adder")?.remove();
    const box = anchor.getBoundingClientRect();
    const menu = h("div", { class: "fl-adder", style: `left:${box.left}px; top:${box.bottom + 6}px` });

    let group = null;
    for (const [kind, spec] of Object.entries(KINDS)) {
      if (spec.group !== group) {
        group = spec.group;
        menu.append(h("div", { class: "fl-group" }, group));
      }
      menu.append(h("button", { onclick: () => { addStep(kind); menu.remove(); } },
        spec.label, h("span", { class: "fl-what" }, spec.what)));
    }
    document.body.append(menu);
    // Closing on the next click anywhere, registered after this one has
    // finished, or the click that opened it would immediately close it.
    setTimeout(() => document.addEventListener("click", function away() {
      menu.remove();
      document.removeEventListener("click", away);
    }), 0);
  }

  function addStep(kind) {
    const id = freshId(kind);
    // Placed in the middle of what is currently on screen, so a new step
    // appears where the person is looking rather than at the origin.
    const canvas = document.getElementById("canvas").getBoundingClientRect();
    const centre = {
      x: (canvas.width / 2 - view.pan.x) / view.zoom - 104,
      y: (canvas.height / 2 - view.pan.y) / view.zoom - 36,
    };
    view.flow.nodes.push({
      id, kind, name: KINDS[kind].label,
      x: Math.round(centre.x + (view.flow.nodes.length % 4) * 24),
      y: Math.round(centre.y + (view.flow.nodes.length % 4) * 24),
      params: {},
    });
    touch();
    select(id);
  }

  /** A short, stable id — this is what expressions refer to. */
  function freshId(kind) {
    const stem = { manual: "start", schedule: "timer", webhook: "hook", agent: "ask", tool: "use", condition: "check", text: "text" }[kind] || "step";
    const taken = new Set(view.flow.nodes.map((n) => n.id));
    if (!taken.has(stem)) return stem;
    for (let i = 2; ; i++) if (!taken.has(`${stem}${i}`)) return `${stem}${i}`;
  }

  // ------------------------------------------------------------ inspector

  function showInspector() {
    const panel = document.getElementById("inspector");
    if (!panel) return;
    const node = view.flow.nodes.find((n) => n.id === view.selected);
    if (!node) { panel.replaceChildren(); return; }

    /// `rebuild` is for settings that change which *other* settings exist —
    /// choosing a tool, which brings its own arguments with it. Everything
    /// else redraws only the card, so typing in a field does not tear down
    /// the field being typed in.
    const set = (key, value, rebuild = false) => {
      node.params = node.params || {};
      if (value === "" || value == null) delete node.params[key];
      else node.params[key] = value;
      touch();
      redrawStep(node);
      if (rebuild) showInspector();
    };

    const fields = [
      field("Name", h("input", {
        value: node.name || "",
        oninput: (e) => { node.name = e.target.value; touch(); redrawStep(node); },
      }), null, `refer to it as ${node.id}`),
      ...settingsFor(node, set),
    ];

    panel.replaceChildren(
      h("h2", {}, KINDS[node.kind]?.label || node.kind),
      ...fields,
      h("button", { class: "ghost-btn auto fl-remove", onclick: () => removeStep(node.id) }, "Remove step"));
  }

  function field(label, control, hint, ref) {
    return h("div", { class: "fl-field" },
      h("label", {}, h("span", {}, label), ref ? h("span", { class: "fl-ref" }, ref) : null),
      control,
      hint ? h("div", { class: "fl-hint" }, hint) : null);
  }

  function settingsFor(node, set) {
    const p = node.params || {};
    switch (node.kind) {
      case "manual":
        return [h("div", { class: "fl-field" },
          h("div", { class: "fl-hint" }, "Press Run above. Nothing else starts this."))];

      case "schedule":
        return [
          field("Every", h("input", {
            value: p.every || "", placeholder: "15m",
            oninput: (e) => set("every", e.target.value.trim()),
          }), "30s, 5m, 2h, 1d. Half a minute is the shortest allowed."),
          field("Or daily at", h("input", {
            value: p.at || "", placeholder: "09:00",
            oninput: (e) => set("at", e.target.value.trim()),
          }), "In UTC, not your local time. A time here wins over an interval."),
          h("div", { class: "fl-field" },
            h("div", { class: "fl-hint" }, "Switch on ", h("strong", {}, "Live"), " above, or nothing fires.")),
        ];

      case "webhook":
        return [
          field("URL", h("div", { class: "fl-hook" }, `${location.origin}/hooks/${view.uuid}/${node.id}`),
            "POST anything JSON here. The body is what this step outputs."),
          h("div", { class: "fl-field" },
            h("div", { class: "fl-hint" }, "It only answers while ", h("strong", {}, "Live"), " is on.")),
        ];

      case "agent": {
        const models = view.palette.models || [];
        return [
          field("Prompt", h("textarea", {
            rows: 7, value: p.prompt || "",
            oninput: (e) => set("prompt", e.target.value),
          }), refHint()),
          field("Model", h("select", { onchange: (e) => set("model", e.target.value) },
            h("option", { value: "", ...(p.model ? {} : { selected: true }) }, "the default"),
            models.map((m) => h("option", { value: m, ...(p.model === m ? { selected: true } : {}) }, m))),
            "Outputs { text }."),
        ];
      }

      case "tool": {
        const tools = view.palette.tools || [];
        const chosen = tools.find((t) => t.name === p.tool);
        const out = [
          field("Tool", h("select", { onchange: (e) => set("tool", e.target.value, true) },
            h("option", { value: "" }, "choose one"),
            tools.map((t) => h("option", {
              value: t.name, ...(p.tool === t.name ? { selected: true } : {}),
            }, `${t.name} — ${t.effect}`))),
            chosen?.description),
        ];
        if (chosen?.blocked) out.push(h("div", { class: "fl-blocked" }, chosen.blocked));
        if (chosen) out.push(...toolArguments(chosen, node, set));
        return out;
      }

      case "condition":
        return [
          field("This", h("input", {
            value: p.left || "", placeholder: "{{ start.status }}",
            oninput: (e) => set("left", e.target.value),
          }), refHint()),
          field("Is", h("select", { onchange: (e) => set("op", e.target.value) },
            OPS.map(([k, label]) => h("option", {
              value: k, ...((p.op || "eq") === k ? { selected: true } : {}),
            }, label)))),
          field("That", h("input", {
            value: p.right || "", placeholder: "ok",
            oninput: (e) => set("right", e.target.value),
          }), "Numbers and the same number written as text count as equal."),
        ];

      case "text":
        return [field("Text", h("textarea", {
          rows: 8, value: p.template || "",
          oninput: (e) => set("template", e.target.value),
        }), refHint())];

      default:
        return [];
    }
  }

  const refHint = () =>
    "Use {{ step.field }} to pull in what an earlier step produced — the step's " +
    "id is shown next to its name.";

  /** A form for a tool, built from the schema the tool already declares. */
  function toolArguments(tool, node, set) {
    const schema = tool.input_schema || {};
    const properties = schema.properties || {};
    const required = new Set(schema.required || []);
    const args = node.params?.arguments || {};

    const write = (key, value) => {
      const next = { ...(node.params?.arguments || {}) };
      if (value === "" || value == null) delete next[key];
      else next[key] = value;
      set("arguments", next);
    };

    return Object.entries(properties).map(([key, spec]) => {
      const value = args[key];
      const type = spec.type === "integer" || spec.type === "number" ? "number" : "text";
      const label = required.has(key) ? `${key} *` : key;

      if (spec.type === "boolean") {
        return field(label, h("select", { onchange: (e) => write(key, e.target.value === "" ? "" : e.target.value === "true") },
          h("option", { value: "", ...(value == null ? { selected: true } : {}) }, "not set"),
          h("option", { value: "true", ...(value === true ? { selected: true } : {}) }, "true"),
          h("option", { value: "false", ...(value === false ? { selected: true } : {}) }, "false")),
          spec.description);
      }
      if (Array.isArray(spec.enum)) {
        return field(label, h("select", { onchange: (e) => write(key, e.target.value) },
          h("option", { value: "" }, "not set"),
          spec.enum.map((option) => h("option", {
            value: option, ...(value === option ? { selected: true } : {}),
          }, String(option)))), spec.description);
      }
      return field(label, h("input", {
        type: value != null && typeof value === "string" && value.includes("{{") ? "text" : type,
        value: value == null ? "" : String(value),
        placeholder: spec.default != null ? String(spec.default) : "",
        // A number field holding `{{ start.count }}` is not a number, so
        // anything with a reference in it stays text and the server resolves
        // it back to a number when the run happens.
        oninput: (e) => {
          const raw = e.target.value;
          const numeric = type === "number" && raw !== "" && !raw.includes("{{");
          write(key, numeric ? Number(raw) : raw);
        },
      }), spec.description);
    });
  }

  function redrawStep(node) {
    const el = document.querySelector(`.fl-step[data-id="${CSS.escape(node.id)}"]`);
    if (!el) return draw();
    el.replaceWith(stepEl(node));
    drawWires();
    showProblems();
  }

  // --------------------------------------------------------- saving, running

  let saveTimer = null;

  function touch() {
    view.dirty = true;
    const slot = document.getElementById("flow-saved");
    if (slot) slot.textContent = "unsaved";
    // Autosaved, because a canvas is fiddled with continuously and losing a
    // layout to a closed tab is the kind of thing that stops people using it.
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 1200);
  }

  async function save() {
    if (!view.flow || view.saving) return;
    clearTimeout(saveTimer);
    view.saving = true;
    const slot = document.getElementById("flow-saved");
    try {
      const saved = await call(`/api/flows/${view.id}`, {
        method: "PUT",
        body: JSON.stringify({ definition: view.flow, enabled: view.enabled }),
      });
      view.dirty = false;
      view.enabled = saved.enabled;
      const toggle = document.getElementById("flow-enabled");
      // The server refuses to arm a flow that cannot run; the checkbox has to
      // agree with it, or it looks armed and silently is not.
      if (toggle) toggle.checked = saved.enabled;
      if (slot) slot.textContent = "saved";
    } catch (e) {
      if (slot) slot.textContent = `not saved: ${e.message}`;
    } finally {
      view.saving = false;
    }
  }

  async function runFlow() {
    if (view.dirty) await save();
    view.run = {};
    draw();
    showRunLog({ live: true, steps: [] });

    const button = document.getElementById("run-flow");
    button.disabled = true;
    button.textContent = "Running…";

    const rows = [];
    try {
      const res = await fetch(`/api/flows/${view.id}/run`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);

      for await (const event of sse(res)) {
        if (event.type === "started") {
          view.run[event.id] = { status: "running" };
          draw();
        } else if (event.type === "finished") {
          const step = event.step;
          view.run[step.id] = {
            status: step.status,
            ms: step.ms,
            error: step.error,
            port: portTaken(step),
          };
          rows.push(step);
          showRunLog({ live: true, steps: rows });
          draw();
        } else if (event.type === "done") {
          for (const step of event.run.steps) {
            view.run[step.id] = {
              status: step.status, ms: step.ms, error: step.error, port: portTaken(step),
            };
          }
          showRunLog({ live: false, steps: event.run.steps, ms: event.run.ms, status: event.run.status });
          draw();
        } else if (event.type === "refused") {
          showRunLog({ live: false, steps: [], refused: event.message });
        }
      }
    } catch (e) {
      showRunLog({ live: false, steps: rows, refused: e.message });
    } finally {
      button.disabled = false;
      button.textContent = "Run";
    }
  }

  /** Which way a condition sent the run, so the wire can show it. */
  function portTaken(step) {
    if (step.kind !== "condition") return "out";
    return step.output?.met ? "yes" : "no";
  }

  async function *sse(res) {
    const reader = res.body.getReader();
    const decode = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decode.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          try { yield JSON.parse(line.slice(5).trim()); } catch {}
        }
      }
    }
  }

  const MARKS = { ran: "✓", failed: "✗", skipped: "·", running: "●" };

  function showRunLog({ live, steps, ms, status, refused }) {
    document.querySelector(".fl-runlog")?.remove();
    const body = document.getElementById("flow-body");
    if (!body) return;

    body.append(h("div", { class: "fl-runlog" },
      h("header", {},
        h("span", {}, refused ? "Could not run" : live ? "Running" : status === "failed" ? "Finished with a failure" : "Finished"),
        ms != null ? h("span", { class: "fl-when" }, `${ms}ms`) : null,
        h("span", { class: "fl-grow" }),
        h("button", {
          class: "icon-btn",
          onclick: () => { document.querySelector(".fl-runlog").remove(); view.run = null; draw(); },
        }, "×")),
      h("div", { class: "fl-rows" },
        refused ? h("div", { class: "fl-runrow fl-failed" },
          h("span", { class: "fl-mark" }, MARKS.failed), h("span", {}, refused)) : null,
        steps.map((s) => h("div", { class: `fl-runrow fl-${s.status}` },
          h("span", { class: "fl-mark" }, MARKS[s.status] || ""),
          h("span", {}, s.label),
          h("span", { class: "fl-ms" }, s.status === "skipped" ? "not reached" : `${s.ms}ms`),
          s.error ? h("span", { class: "fl-err" }, s.error) : null)))));
  }

  async function loadRuns() {
    try {
      const runs = await call(`/api/flows/${view.id}/runs`);
      if (!runs.length) return;
      const last = runs[0];
      showRunLog({ live: false, steps: last.record?.steps || [], ms: last.ms, status: last.status });
    } catch {}
  }

  // -------------------------------------------------------------- routing

  function go(path) {
    history.pushState({}, "", path);
    open();
  }

  /** Whether this route belongs to workflows. Called by the chat app's router. */
  function owns(pathname) {
    return pathname === "/flows" || pathname.startsWith("/flows/");
  }

  async function open() {
    const host = document.getElementById("flows-host");
    const chat = document.querySelector(".main:not(.flows-host)");
    if (!host) return;
    host.hidden = false;
    if (chat) chat.hidden = true;
    document.getElementById("open-flows")?.classList.add("current");
    const match = location.pathname.match(/^\/flows\/(\d+)$/);
    try {
      if (match) await showEditor(host, Number(match[1]));
      else await showList(host);
    } catch (e) {
      host.replaceChildren(h("div", { class: "fl-flow-list" },
        h("h1", {}, "Workflows"),
        h("p", { class: "fl-lede" }, `Could not load: ${e.message}`)));
    }
  }

  /** Put the chat back. Called by the chat app when it takes a route. */
  function close() {
    const host = document.getElementById("flows-host");
    const chat = document.querySelector(".main:not(.flows-host)");
    if (host && !host.hidden) {
      host.hidden = true;
      host.replaceChildren();
      document.removeEventListener("keydown", onKey);
      view.flow = null;
    }
    if (chat) chat.hidden = false;
    document.getElementById("open-flows")?.classList.remove("current");
  }

  window.Flows = { owns, open, close, go };
})();
