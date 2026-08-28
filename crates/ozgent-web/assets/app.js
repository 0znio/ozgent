// ozgent web client.
//
// Deliberately dependency-free: the page is served from the binary, so every
// byte here ships inside it and a bundler would buy nothing.

const $ = (id) => document.getElementById(id);
const el = {
  thread: $("thread"), input: $("input"), stat: $("stat"), model: $("model"),
  thinking: $("thinking"), convs: $("conversations"), empty: $("empty"),
  send: $("send"), sidebar: $("sidebar"), attachments: $("attachments"),
  quick: $("quick"),
};

// The handful of settings worth reaching for mid-conversation. Everything else
// lives in Settings > Model; putting it all here would defeat the point.
const QUICK = [
  { key: "temperature", label: "Temperature", min: 0, max: 2, step: 0.05 },
  { key: "top_p", label: "Top-p", min: 0, max: 1, step: 0.01 },
  { key: "top_k", label: "Top-k", min: 1, max: 200, step: 1 },
  { key: "max_tokens", label: "Max output", min: 0, max: 4096, step: 64, zero: "model max" },
  { key: "context_length", label: "Context", min: 512, max: 32768, step: 512 },
];

/// Widen a slider to the model's own limit, and keep the step usable.
///
/// The ceiling used to be the *current* value, so a model trained for 256k got
/// a slider that stopped at the 32k written in this file. The limit now comes
/// from the GGUF, and the step scales with it: 512-token increments across a
/// 262,144-token range is 512 positions of a mouse drag nobody wants.
function bound(spec, limits) {
  const max = Number(limits?.[spec.key]) || 0;
  if (!max || max <= spec.max) return spec;
  // Around 128 positions, rounded to a whole number of 512-token blocks, so
  // the value always lands somewhere a model would actually be configured to.
  const step = Math.max(spec.step, Math.round(max / 128 / 512) * 512 || spec.step);
  return { ...spec, max, step };
}

// Not a slider: effort is three named settings, not a continuum, and showing
// it as one would invite a 0.37 that means nothing.
const CHOICES = [
  {
    key: "reasoning_effort",
    label: "Reasoning",
    options: ["low", "medium", "high"],
    hint: "How long a thinking model may reason before it answers",
  },
];

const state = {
  conversation: null, streaming: false, abort: null, files: [], tools: true,
  /// Row id to public id, so a route can be written without another request.
  uuids: new Map(),
};

// ----------------------------------------------------------------- theme

// Three states, not two: "system" is the one most people want and a two-way
// switch cannot express it — once flipped, the page stops following the
// desktop for good. The stylesheet already reads these stamps; all this does
// is set them and remember which.
const THEMES = [
  { id: "system", icon: "i-monitor", name: "System" },
  { id: "light", icon: "i-sun", name: "Light" },
  { id: "dark", icon: "i-moon", name: "Dark" },
];

/// Point an inline icon at a different symbol in the sprite.
function setIcon(svg, symbol) {
  svg?.querySelector("use")?.setAttribute("href", `#${symbol}`);
}

function applyTheme(id) {
  const theme = THEMES.find((t) => t.id === id) ?? THEMES[0];
  // "system" removes the stamp rather than setting one, which is what lets
  // `prefers-color-scheme` decide again.
  if (theme.id === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme.id);

  const button = $("theme-toggle");
  if (button) {
    setIcon(button.querySelector(".theme-icon"), theme.icon);
    button.querySelector(".theme-name").textContent = theme.name;
    button.title = `Theme: ${theme.name} (click to change)`;
  }
  // Private browsing and blocked site data both make this throw; the theme
  // still applies, it just will not be remembered.
  try { localStorage.setItem("ozgent-theme", theme.id); } catch (_) { /* not fatal */ }
  return theme.id;
}

function storedTheme() {
  try { return localStorage.getItem("ozgent-theme") ?? "system"; } catch (_) { return "system"; }
}

// ------------------------------------------------------------- utilities

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).error ?? detail; } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

/// A coarse "when", accurate enough to tell two threads apart in a list.
///
/// Matches the terminal client's wording, so the same conversation reads the
/// same in both. A clock ahead of the database lands on "just now" rather than
/// a negative duration.
function ago(seconds) {
  const now = Math.floor(Date.now() / 1000);
  const d = Math.max(0, now - (seconds || 0));
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  if (d < 86400 * 30) return `${Math.floor(d / 86400)}d ago`;
  return `${Math.floor(d / (86400 * 30))}mo ago`;
}

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Enough markdown for model output: fences, inline code, headings, lists,
// emphasis, links, tables, quotes and rules. Anything richer is not worth a
// parser on the client.
//
// Fenced code is lifted out before anything else runs and put back last. The
// marker it leaves behind is a control character rather than a run of spaces:
// the block loop trims each line, which ate the old ` 0 ` marker's trailing
// space, so the pattern never matched again and every code block was replaced
// by a bare digit. Streaming made that look like code appearing as plain text
// and then vanishing the instant its closing fence arrived.
const MARK = "\u0001";

function markdown(src) {
  const blocks = [];
  const stash = (html) => {
    blocks.push(html);
    return `${MARK}${blocks.length - 1}${MARK}`;
  };
  const codeBlock = (lang, code, open = false) =>
    `<pre${open ? ' data-unterminated="1"' : ""}>` +
    `<code class="lang-${escapeHtml(lang || "text")}">` +
    `${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;

  // The marker must not occur in the model's own output.
  src = src.split(MARK).join("");

  src = src.replace(/```([\w+-]*)[ \t]*\n?([\s\S]*?)```/g, (_, lang, code) =>
    stash(codeBlock(lang, code)),
  );
  // An unterminated fence is the normal state mid-stream. Rendering the rest
  // as code shows it forming, rather than as unstyled text that reflows the
  // moment the closing fence lands.
  // Marked, because the same shape means two different things. Mid-stream the
  // closing fence has not arrived yet; in a finished reply the model forgot
  // it, and everything written afterwards is now inside the block. Guessing
  // where it meant to stop would cut real code in half, so the block is
  // labelled rather than repaired.
  src = src.replace(/```([\w+-]*)[ \t]*\n?([\s\S]*)$/, (_, lang, code) =>
    stash(codeBlock(lang, code, true)),
  );

  const inline = (t) => {
    const spans = [];
    // Inline code is protected the same way, or emphasis inside `a*b*c` would
    // rewrite characters the span exists to show literally.
    let text = t.replace(/(`+)([\s\S]*?)\1/g, (_, _ticks, code) => {
      spans.push(`<code>${escapeHtml(code.trim())}</code>`);
      return `${MARK}s${spans.length - 1}${MARK}`;
    });
    text = escapeHtml(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/~~([^~]+)~~/g, "<del>$1</del>")
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>")
      .replace(
        /\[([^\]]*)\]\((https?:[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
      );
    return text.replace(
      new RegExp(`${MARK}s(\\d+)${MARK}`, "g"),
      (_, n) => spans[Number(n)],
    );
  };

  // Tables need a line of lookahead — a row is only a table row if the line
  // after it is a separator — so this walks by index rather than for..of.
  const lines = src.split("\n");
  const out = [];
  // Open list elements, outermost first, so nesting by indent closes to the
  // right depth rather than all at once.
  const lists = [];
  const closeLists = (toDepth = 0) => {
    while (lists.length > toDepth) out.push(`</${lists.pop().tag}>`);
  };

  const cellsOf = (line) => {
    let t = line.trim();
    if (t.startsWith("|")) t = t.slice(1);
    if (t.endsWith("|")) t = t.slice(0, -1);
    return t.split("|").map((c) => c.trim());
  };
  const isSeparator = (line) =>
    line !== undefined && /\|/.test(line) && /-/.test(line) && /^[\s|:-]+$/.test(line);
  const alignOf = (cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    if (left && right) return "center";
    if (right) return "right";
    return left ? "left" : "";
  };

  let quote = [];
  const flushQuote = () => {
    if (!quote.length) return;
    out.push(`<blockquote>${markdown(quote.join("\n"))}</blockquote>`);
    quote = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trimEnd();

    const placeholder = line.trim().match(new RegExp(`^${MARK}(\\d+)${MARK}$`));
    if (placeholder) {
      flushQuote();
      closeLists();
      out.push(blocks[Number(placeholder[1])]);
      continue;
    }

    const quoted = line.match(/^\s{0,3}>\s?(.*)$/);
    if (quoted) {
      closeLists();
      quote.push(quoted[1]);
      continue;
    }
    flushQuote();

    // A thematic break: three or more of -, * or _, optionally spaced.
    // Checked before lists, or `- - -` is read as a bullet, and before
    // paragraphs, where `---` used to end up as its literal characters.
    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      closeLists();
      out.push("<hr>");
      continue;
    }

    // GFM table: a header row, a separator, then body rows.
    if (line.includes("|") && isSeparator(lines[i + 1])) {
      closeLists();
      const head = cellsOf(line);
      const aligns = cellsOf(lines[i + 1]).map(alignOf);
      const rows = [];
      let k = i + 2;
      while (k < lines.length && lines[k].includes("|") && lines[k].trim()) {
        rows.push(cellsOf(lines[k]));
        k++;
      }
      const cell = (tag, text, n) => {
        const a = aligns[n] ? ` style="text-align:${aligns[n]}"` : "";
        return `<${tag}${a}>${inline(text)}</${tag}>`;
      };
      out.push(
        '<div class="table-wrap"><table><thead><tr>' +
          head.map((h, n) => cell("th", h, n)).join("") +
          "</tr></thead><tbody>" +
          rows
            .map(
              (r) =>
                "<tr>" +
                head.map((_, n) => cell("td", r[n] ?? "", n)).join("") +
                "</tr>",
            )
            .join("") +
          "</tbody></table></div>",
      );
      i = k - 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeLists();
      // Offset so a model's `#` is a section inside the reply rather than a
      // title competing with the page's own; capped at h6, the last one.
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = raw.match(/^(\s*)[-*+]\s+(.*)$/);
    const number = raw.match(/^(\s*)\d+[.)]\s+(.*)$/);
    if (bullet || number) {
      const [, indent, text] = bullet || number;
      const tag = bullet ? "ul" : "ol";
      // Two spaces to a level, which is what a model writes; four lands on a
      // level too, so both conventions nest rather than one flattening.
      const depth = Math.floor(indent.replace(/\t/g, "  ").length / 2);

      while (lists.length > depth + 1) out.push(`</${lists.pop().tag}>`);
      if (lists.length === depth + 1 && lists[depth].tag !== tag) {
        out.push(`</${lists.pop().tag}>`);
      }
      while (lists.length < depth + 1) {
        out.push(`<${tag}>`);
        lists.push({ tag });
      }
      out.push(`<li>${inline(text)}</li>`);
      continue;
    }

    if (!line.trim()) { closeLists(); continue; }
    closeLists();
    out.push(`<p>${inline(line)}</p>`);
  }
  flushQuote();
  closeLists();
  return out.join("\n");
}

// ------------------------------------------------------- syntax colour
//
// A hand-rolled tokenizer, because the alternative is a CDN and this page has
// to render with no network at all. It is small on purpose: the job is to make
// code read as code — comments receding, strings and keywords separating from
// identifiers — not to be a language server. Anything it does not know falls
// back to plain monospace, which is what the block looked like before.
//
// It scans the *text*, never the markup, and escapes on the way out. A
// highlighter that rewrites already-escaped HTML with regexes is how you turn
// a code block into an injection point.

/// What a language is made of. `kw` and `typ` are two weights of the same
/// idea: the words that steer control flow, and the words that name things
/// the language provides.
const LANG_SPEC = (() => {
  const cLike = { line: ["//"], block: [["/*", "*/"]], quotes: ['"', "'", "`"] };
  const hash = { line: ["#"], block: [], quotes: ['"', "'"] };

  const spec = {
    python: {
      ...hash,
      // Triple quotes first, or the scanner closes on the opening pair.
      quotes: ['"""', "'''", '"', "'"],
      kw: `and as assert async await break class continue def del elif else except finally
           for from global if import in is lambda nonlocal not or pass raise return try
           while with yield match case`,
      typ: `True False None self cls int float str bool list dict set tuple bytes object
            len range print open enumerate zip map filter sum min max abs sorted isinstance
            super type Exception ValueError TypeError KeyError IndexError`,
    },
    javascript: {
      ...cLike,
      kw: `async await break case catch class const continue debugger default delete do else
           export extends finally for from function get if import in instanceof let new of
           return set static super switch this throw try typeof var void while with yield`,
      typ: `true false null undefined NaN Infinity console window document Math JSON Object
            Array String Number Boolean Promise Map Set Symbol Error RegExp Date globalThis`,
    },
    typescript: null, // filled in below
    rust: {
      ...cLike,
      quotes: ['"'],
      kw: `as async await break const continue crate dyn else enum extern fn for if impl in
           let loop match mod move mut pub ref return self Self static struct super trait
           type unsafe use where while`,
      typ: `bool char f32 f64 i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize str String
            Vec Option Some None Result Ok Err Box Rc Arc HashMap HashSet true false`,
    },
    go: {
      ...cLike,
      kw: `break case chan const continue default defer else fallthrough for func go goto if
           import interface map package range return select struct switch type var`,
      typ: `bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64
            rune string uint uint8 uint16 uint32 uint64 uintptr true false nil iota make new
            len cap append copy delete panic recover`,
    },
    c: {
      ...cLike,
      quotes: ['"', "'"],
      kw: `auto break case const continue default do else enum extern for goto if inline
           register restrict return sizeof static struct switch typedef union volatile while
           class namespace template public private protected virtual override new delete
           using try catch throw constexpr nullptr`,
      typ: `bool char double float int long short signed unsigned void size_t NULL true false
            std string vector map set auto uint8_t uint32_t int32_t int64_t`,
    },
    java: {
      ...cLike,
      quotes: ['"', "'"],
      kw: `abstract assert break case catch class const continue default do else enum extends
           final finally for goto if implements import instanceof interface native new package
           private protected public return static strictfp super switch synchronized this
           throw throws transient try void volatile while`,
      typ: `boolean byte char double float int long short String Object List Map Set Integer
            Double Boolean Long System true false null var record sealed`,
    },
    bash: {
      ...hash,
      quotes: ['"', "'"],
      kw: `if then else elif fi for while until do done case esac function select in return
           break continue exit local export readonly declare shift trap set unset source eval`,
      typ: `echo cd ls cat grep sed awk find cp mv rm mkdir touch chmod chown curl wget git
            sudo apt yum pip python python3 node npm cargo make docker kubectl test true false`,
    },
    sql: {
      line: ["--"], block: [["/*", "*/"]], quotes: ["'", '"'],
      kw: `select from where insert into values update set delete create table alter drop
           index view join inner left right outer full on group by order having limit offset
           union all distinct as and or not null is in exists between like case when then
           else end primary key foreign references default constraint unique begin commit
           rollback transaction with returning`,
      typ: `int integer bigint smallint serial text varchar char boolean bool date timestamp
            timestamptz numeric decimal real double float json jsonb uuid array count sum avg
            min max coalesce now true false`,
      fold: true,
    },
    json: { line: [], block: [], quotes: ['"'], kw: `true false null`, typ: `` },
    yaml: { ...hash, kw: `true false null yes no on off`, typ: `` },
    toml: { ...hash, kw: `true false`, typ: `` },
    css: {
      line: [], block: [["/*", "*/"]], quotes: ['"', "'"],
      kw: `import media supports keyframes font-face charset use include mixin extend
           if else for each while return`,
      typ: `inherit initial unset auto none block flex grid inline absolute relative fixed
            sticky hidden visible solid dashed dotted transparent currentColor var calc rgb
            rgba hsl url`,
    },
    ruby: {
      ...hash,
      kw: `alias and begin break case class def defined do else elsif end ensure false for if
           in module next nil not or redo rescue retry return self super then true undef
           unless until when while yield require require_relative attr_accessor`,
      typ: `puts print p String Integer Float Array Hash Symbol Proc Struct nil true false`,
    },
  };

  spec.typescript = {
    ...spec.javascript,
    kw: `${spec.javascript.kw} abstract as any declare enum implements interface is keyof
         namespace never private protected public readonly satisfies type unknown`,
    typ: `${spec.javascript.typ} string number boolean object bigint symbol Record Partial
          Readonly Pick Omit Array`,
  };

  // Turn the word lists into sets once, rather than splitting on every token.
  for (const key of Object.keys(spec)) {
    const s = spec[key];
    if (!s) continue;
    s.kwSet = new Set((s.kw || "").split(/\s+/).filter(Boolean));
    s.typSet = new Set((s.typ || "").split(/\s+/).filter(Boolean));
  }
  return spec;
})();

/// Map what a model writes in a fence to a spec.
const LANG_ALIAS = {
  py: "python", python3: "python", js: "javascript", mjs: "javascript", cjs: "javascript",
  jsx: "javascript", node: "javascript", ts: "typescript", tsx: "typescript",
  rs: "rust", sh: "bash", shell: "bash", zsh: "bash", console: "bash", terminal: "bash",
  golang: "go", "c++": "c", cpp: "c", cc: "c", h: "c", hpp: "c", cs: "java", csharp: "java",
  kotlin: "java", kt: "java", swift: "java", scala: "java", postgres: "sql", psql: "sql",
  mysql: "sql", sqlite: "sql", yml: "yaml", scss: "css", sass: "css", less: "css", rb: "ruby",
};

const specFor = (lang) => {
  const key = String(lang || "").toLowerCase();
  return LANG_SPEC[LANG_ALIAS[key] || key] || null;
};

const IDENT_START = /[A-Za-z_$@]/;
const IDENT_REST = /[A-Za-z0-9_$-]/;

/// Colour one block of source.
///
/// Returns escaped HTML. Every branch escapes what it emits, so a `<script>`
/// in a string is shown, never run.
function highlight(src, lang) {
  const spec = specFor(lang);
  if (!spec) return escapeHtml(src);

  const out = [];
  const push = (cls, text) =>
    out.push(cls ? `<span class="t-${cls}">${escapeHtml(text)}</span>` : escapeHtml(text));

  let i = 0;
  const n = src.length;
  let plain = "";
  const flush = () => { if (plain) { push(null, plain); plain = ""; } };

  while (i < n) {
    const rest = src.slice(i);

    // --- comments ---
    const line = spec.line.find((m) => rest.startsWith(m));
    if (line) {
      flush();
      const end = src.indexOf("\n", i);
      const stop = end === -1 ? n : end;
      push("com", src.slice(i, stop));
      i = stop;
      continue;
    }
    const block = spec.block.find(([open]) => rest.startsWith(open));
    if (block) {
      flush();
      const close = src.indexOf(block[1], i + block[0].length);
      const stop = close === -1 ? n : close + block[1].length;
      push("com", src.slice(i, stop));
      i = stop;
      continue;
    }

    // --- strings ---
    // Longest opener first, so `"""` is not read as `"` followed by `""`.
    const quote = [...spec.quotes].sort((a, b) => b.length - a.length)
      .find((q) => rest.startsWith(q));
    if (quote) {
      flush();
      const multi = quote.length > 1;
      let j = i + quote.length;
      while (j < n) {
        if (src[j] === "\\") { j += 2; continue; }
        if (src.startsWith(quote, j)) { j += quote.length; break; }
        // A single-quoted string that runs off the line was never a string;
        // stopping here keeps one stray apostrophe from colouring the rest of
        // the file.
        if (!multi && src[j] === "\n") break;
        j++;
      }
      push("str", src.slice(i, Math.min(j, n)));
      i = Math.min(j, n);
      continue;
    }

    const ch = src[i];

    // --- numbers ---
    if (/[0-9]/.test(ch) && !(i > 0 && IDENT_REST.test(src[i - 1]))) {
      flush();
      const m = rest.match(/^(0[xXbBoO][0-9a-fA-F_]+|[0-9][0-9_]*(\.[0-9_]+)?([eE][+-]?[0-9]+)?)[a-zA-Z_]*/);
      const text = m ? m[0] : ch;
      push("num", text);
      i += text.length;
      continue;
    }

    // --- words ---
    if (IDENT_START.test(ch)) {
      let j = i + 1;
      while (j < n && IDENT_REST.test(src[j])) j++;
      const word = src.slice(i, j);
      const lookup = spec.fold ? word.toLowerCase() : word;

      let cls = null;
      if (spec.kwSet.has(lookup)) cls = "kw";
      else if (spec.typSet.has(lookup)) cls = "typ";
      else {
        // A name immediately followed by `(` is being called. Cheap, and right
        // often enough to be worth the two characters of lookahead.
        let k = j;
        while (k < n && (src[k] === " " || src[k] === "\t")) k++;
        if (src[k] === "(") cls = "fn";
        else if (word[0] === "@" || word[0] === "$") cls = "typ";
      }
      if (cls) { flush(); push(cls, word); } else { plain += word; }
      i = j;
      continue;
    }

    // --- punctuation ---
    if (/[{}()[\];,.:=+\-*/%<>!&|^~?]/.test(ch)) {
      flush();
      push("op", ch);
      i++;
      continue;
    }

    plain += ch;
    i++;
  }
  flush();
  return out.join("");
}

/// Highlighted HTML, remembered by source.
///
/// `dressCode` runs on every streamed token and `innerHTML` rebuilds all the
/// blocks each time, so without this a reply with five code blocks would
/// re-colour all five on every token. Only the block still being written
/// misses, which bounds the work to the tail.
const HIGHLIGHT_CACHE = new Map();
const HIGHLIGHT_CACHE_MAX = 120;

function highlightCached(src, lang) {
  // A NUL cannot occur in a language name — the fence regex limits it to word
  // characters — so the two halves cannot be confused for one another.
  const key = `${lang}\u0000${src}`;
  const hit = HIGHLIGHT_CACHE.get(key);
  if (hit !== undefined) {
    // Re-inserting moves it to the end, which turns insertion order into
    // recency. Left as a plain FIFO, streaming one long block writes a new key
    // per token and evicts the finished blocks above it — which are exactly
    // the entries the cache exists to keep.
    HIGHLIGHT_CACHE.delete(key);
    HIGHLIGHT_CACHE.set(key, hit);
    return hit;
  }

  const html = highlight(src, lang);
  if (HIGHLIGHT_CACHE.size >= HIGHLIGHT_CACHE_MAX) {
    HIGHLIGHT_CACHE.delete(HIGHLIGHT_CACHE.keys().next().value);
  }
  HIGHLIGHT_CACHE.set(key, html);
  return html;
}

/// Give every code block a header with its language and a copy control.
///
/// Applied after rendering rather than inside `markdown`, so the renderer
/// stays a pure string function and the DOM work happens once per block
/// instead of on every streamed token.
function dressCode(root) {
  for (const pre of root.querySelectorAll("pre")) {
    if (pre.parentElement?.classList.contains("code-block")) continue;

    const code = pre.querySelector("code");
    const lang = (code?.className.match(/lang-([\w+-]+)/) || [, "text"])[1];
    const open = pre.dataset.unterminated === "1";

    const block = document.createElement("div");
    block.className = "code-block";

    // Colour it. `textContent` is the original source — the markup around it
    // was written by `markdown`, so this never re-parses escaped HTML.
    if (code) {
      const source = code.textContent ?? "";
      const coloured = highlightCached(source, lang);
      if (coloured !== escapeHtml(source)) {
        code.innerHTML = coloured;
        block.classList.add("lit");
      }
    }
    const head = document.createElement("div");
    head.className = "code-head";
    head.innerHTML = '<span class="nm"></span><span class="spacer"></span>' +
      '<button type="button" class="copy-btn" aria-label="Copy this code">' +
      '<svg class="ic"><use href="#i-copy"></use></svg><span>Copy</span></button>';
    head.querySelector(".nm").textContent = lang;
    if (open) {
      block.classList.add("unterminated");
      const flag = document.createElement("span");
      flag.className = "flag";
      flag.textContent = "unclosed";
      flag.title =
        "The model did not close this code block, so everything below is inside it";
      head.querySelector(".nm").after(flag);
    }

    pre.replaceWith(block);
    block.append(head, pre);

    head.querySelector(".copy-btn").addEventListener("click", async (e) => {
      const button = e.currentTarget;
      const ok = await copyText(code?.textContent ?? pre.textContent ?? "");
      // The label is the feedback; a toast for something this small is noise.
      button.querySelector("span").textContent = ok ? "Copied" : "Press Ctrl+C";
      setIcon(button.querySelector(".ic"), ok ? "i-check" : "i-copy");
      button.classList.toggle("done", ok);
      setTimeout(() => {
        button.querySelector("span").textContent = "Copy";
        setIcon(button.querySelector(".ic"), "i-copy");
        button.classList.remove("done");
      }, 1600);
    });
  }
}

/// Copy to the clipboard, falling back where the async API is unavailable.
///
/// `navigator.clipboard` needs a secure context, and reaching this page over
/// plain http from another machine on the network is exactly the case ozgent
/// now supports — so the fallback is the normal path there, not an edge case.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    try {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    } catch (_) {
      return false;
    }
  }
}

// -------------------------------------------------------------- rendering

// The answer lives in its own child so the reasoning block can sit above it
// inside the body. Putting reasoning beside `.body` made it a flex item of
// `.msg`, which rendered it as a narrow column next to the text.
function addMessage(role, text = "") {
  document.getElementById("empty")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML =
    `<div class="who">${role === "user" ? "you" : "oz"}</div>` +
    `<div class="body"><div class="answer"></div></div>`;
  const body = wrap.querySelector(".body");
  const answer = wrap.querySelector(".answer");
  if (role === "user") answer.textContent = text;
  else { answer.innerHTML = markdown(text); dressCode(answer); }
  el.thread.append(wrap);
  scrollToTail();
  return { body, answer };
}

/// Reasoning is rendered as markdown too — models format it, and showing the
/// raw asterisks makes a long trace harder to skim than it needs to be.
function reasoningPane(body) {
  let pane = body.querySelector(".think");
  if (!pane) {
    pane = document.createElement("details");
    pane.className = "think";
    pane.innerHTML = '<summary>reasoning</summary><div class="think-body"></div>';
    body.prepend(pane);
  }
  return pane.querySelector(".think-body");
}

// ------------------------------------------------------------- the gauge

/// How much of the window the conversation is using.
///
/// `done` reports what the turn actually cost — the prompt it processed plus
/// what it generated — which together are the occupancy of the window at the
/// end of that turn. Nothing else in the interface says that a local model has
/// a hard limit on this machine.
const gauge = { window: 0, used: 0 };

const compact = (n) =>
  n >= 1000 ? `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)}k` : String(n);

function showGauge() {
  const box = $("gauge");
  if (!box) return;
  if (!gauge.window || !gauge.used) { box.classList.remove("live"); return; }

  const share = Math.min(1, gauge.used / gauge.window);
  box.classList.add("live");
  // Amber from three quarters on, which is where the oldest turns start
  // being dropped from the context rather than merely getting close.
  box.classList.toggle("warm", share >= 0.75);
  $("gauge-fill").style.width = `${(share * 100).toFixed(1)}%`;
  $("gauge-used").textContent = compact(gauge.used);
  $("gauge-total").textContent = ` / ${compact(gauge.window)}`;
  box.title = `${gauge.used} of ${gauge.window} tokens used`;
}

/// How far the reader is from the newest message, in pixels.
const distanceFromTail = () =>
  el.thread.scrollHeight - el.thread.scrollTop - el.thread.clientHeight;

function scrollToTail() {
  // Only follow the tail if the reader is already near it. Reading back
  // through a long answer must not be yanked forward by the next token.
  if (distanceFromTail() < 160) el.thread.scrollTop = el.thread.scrollHeight;
  syncToLatest();
}

/// Offer a way back to the newest message, but only once it is off screen.
function syncToLatest() {
  const button = $("to-latest");
  if (!button) return;
  const away = distanceFromTail() > 240;
  // `hidden` is removed first so the opacity transition has something to run
  // on; putting it back waits for the fade out.
  if (away) {
    button.hidden = false;
    requestAnimationFrame(() => button.classList.add("show"));
  } else {
    button.classList.remove("show");
  }
}

// One button, two meanings. A separate hidden stop button drifts out of
// alignment and makes the control move under the cursor mid-generation.
function setStreaming(on) {
  state.streaming = on;
  el.send.classList.toggle("is-stop", on);
  setIcon(el.send.querySelector(".ic"), on ? "i-stop" : "i-send");
  el.send.setAttribute("aria-label", on ? "Stop generating" : "Send");
  el.send.type = on ? "button" : "submit";
}

// ------------------------------------------------------------ attachments

/// A tool call shown as one collapsed row that can be opened for the detail.
///
/// The interesting argument leads — for a search that is the query, and it is
/// what a reader scans for — with the rest kept as muted context.
function openToolCard(answerEl, event) {
  const card = document.createElement("details");
  card.className = "tool-card running";
  // Whichever argument names the thing acted on leads. A path identifies a
  // read better than the query does; a search has only its query.
  const args = event.arguments ?? {};
  const leadKey = ["path", "query", "url", "city"].find((k) => args[k] !== undefined);
  const lead = leadKey ? String(args[leadKey]) : "";
  const rest = Object.entries(args)
    .filter(([k]) => k !== leadKey)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ");

  card.innerHTML =
    '<summary>' +
      '<span class="dot"></span>' +
      '<span class="nm"></span>' +
      '<span class="lead"></span>' +
      '<span class="rest"></span>' +
      '<span class="ms">running</span>' +
    '</summary>' +
    '<div class="tool-detail"></div>';
  card.querySelector(".nm").textContent = event.name;
  card.querySelector(".lead").textContent = lead;
  card.querySelector(".rest").textContent = rest;
  answerEl.before(card);
  return card;
}

/// Ask whether a tool call may run.
///
/// The arguments are the question. "Allow run_command?" cannot be answered by
/// anyone — the whole risk is in the string being run — so the card shows the
/// call in full and the buttons are only meaningful underneath it.
///
/// The card answers once and then becomes a record of what was answered. It is
/// never removed: the following tool card or refusal reads as its consequence,
/// and a question that vanishes leaves the transcript saying a tool simply ran.
function permissionCard(answerEl, event) {
  const card = document.createElement("div");
  card.className = `perm perm-${event.effect}`;

  const args = Object.entries(event.arguments ?? {});
  const rows = args.length
    ? args
        .map(
          ([k, v]) =>
            `<div class="perm-arg"><span class="k">${escapeHtml(k)}</span>` +
            `<span class="v">${escapeHtml(typeof v === "string" ? v : JSON.stringify(v))}</span></div>`,
        )
        .join("")
    : '<div class="perm-arg"><span class="v">no arguments</span></div>';

  const what = {
    read: "wants to read something",
    write: "wants to change files or data",
    execute: "wants to run a program",
    unknown: "does not say what it does",
  }[event.effect] ?? "wants to run";

  card.innerHTML =
    '<div class="perm-head">' +
      '<svg class="ic"><use href="#i-shield"></use></svg>' +
      `<span class="nm">${escapeHtml(event.name)}</span>` +
      `<span class="what">${escapeHtml(what)}</span>` +
    '</div>' +
    `<div class="perm-args">${rows}</div>` +
    '<div class="perm-actions">' +
      '<button class="primary-btn" data-choice="once">Allow</button>' +
      '<button class="ghost-btn" data-choice="session">Allow this session</button>' +
      `<button class="ghost-btn" data-choice="always">Always allow ${escapeHtml(event.name)}</button>` +
      '<button class="ghost-btn decline" data-choice="deny">Decline</button>' +
    '</div>' +
    '<div class="perm-outcome" hidden></div>';

  const answered = {
    once: "allowed once",
    session: "allowed for this session",
    always: `always allowed — ${event.name} will not ask again`,
    deny: "declined",
  };

  card.querySelectorAll("[data-choice]").forEach((button) => {
    button.addEventListener("click", async () => {
      const choice = button.dataset.choice;
      // Disabled before the request, not after: a second click would answer a
      // question that is no longer waiting and read as an error.
      card.querySelectorAll("[data-choice]").forEach((b) => (b.disabled = true));
      try {
        await api("/api/permissions/decide", {
          method: "POST",
          body: JSON.stringify({ id: event.id, choice }),
        });
      } catch {
        // A 404 means the turn moved on — the wait ran out, or another tab
        // answered. Nothing to recover, and nothing worth alarming about.
      }
      card.classList.add("answered", choice === "deny" ? "declined" : "allowed");
      card.querySelector(".perm-actions").remove();
      const outcome = card.querySelector(".perm-outcome");
      outcome.textContent = answered[choice];
      outcome.hidden = false;
    });
  });

  answerEl.before(card);
  return card;
}

/// Fill in a card once the tool has returned.
function closeToolCard(card, event) {
  if (!card) return;
  card.classList.remove("running");
  card.classList.toggle("bad", !event.ok);
  card.querySelector(".ms").textContent =
    event.ms >= 1000 ? `${(event.ms / 1000).toFixed(1)}s` : `${event.ms} ms`;

  // The arguments stay: they are what the row is about. Only a failure
  // replaces them, because then the reason is the useful thing to show.
  if (!event.ok) card.querySelector(".rest").textContent = event.summary ?? "failed";

  const detail = card.querySelector(".tool-detail");
  const results = event.detail?.results;
  if (Array.isArray(results) && results.length) {
    const list = document.createElement("ol");
    list.className = "tool-results";
    for (const r of results) {
      const li = document.createElement("li");
      const title = r.title ?? r.url ?? "untitled";
      if (r.url) {
        const a = document.createElement("a");
        a.href = r.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = title;
        li.append(a);
      } else {
        li.append(document.createTextNode(title));
      }
      if (r.snippet ?? r.description) {
        const p = document.createElement("p");
        p.textContent = r.snippet ?? r.description;
        li.append(p);
      }
      list.append(li);
    }
    detail.append(list);
  } else {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(event.detail, null, 2).slice(0, 4000);
    detail.append(pre);
  }
}

/// Arguments on one line, short enough for a transcript row.
function summariseArgs(args) {
  if (!args || typeof args !== "object") return "";
  const parts = Object.entries(args).map(([k, v]) => {
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return `${k}=${s.length > 60 ? s.slice(0, 57) + "..." : s}`;
  });
  return parts.join("  ");
}

const humanSize = (n) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`;

function addFiles(files) {
  for (const f of files) state.files.push(f);
  renderAttachments();
}

function renderAttachments() {
  el.attachments.replaceChildren();
  el.attachments.hidden = state.files.length === 0;
  state.files.forEach((f, i) => {
    const chip = document.createElement("div");
    chip.className = "chip-file";
    const isImage = f.type.startsWith("image/");
    chip.innerHTML =
      (isImage ? `<img alt="">` : "") +
      `<span class="nm"></span><span class="sz"></span><button type="button" aria-label="Remove">&times;</button>`;
    chip.querySelector(".nm").textContent = f.name || "pasted image";
    chip.querySelector(".sz").textContent = humanSize(f.size);
    if (isImage) chip.querySelector("img").src = URL.createObjectURL(f);
    chip.querySelector("button").addEventListener("click", () => {
      state.files.splice(i, 1);
      renderAttachments();
    });
    el.attachments.append(chip);
  });
}

// ---------------------------------------------------------- conversations

async function loadConversations() {
  const list = await api("/api/conversations");
  state.uuids = new Map(list.map((c) => [c.id, c.uuid]));
  el.convs.replaceChildren();
  for (const c of list) {
    const row = document.createElement("div");
    row.className = "conv";
    row.setAttribute("role", "button");
    row.tabIndex = 0;
    if (c.id === state.conversation) row.setAttribute("aria-current", "true");
    row.innerHTML =
      '<span class="conv-text"><span class="title"></span><span class="when"></span></span>' +
      '<button class="del" title="Delete" aria-label="Delete conversation">' +
      '<svg class="ic"><use href="#i-trash"></use></svg></button>';
    row.querySelector(".title").textContent = c.title || "Untitled";
    // When it was last touched, and how long it ran. Enough to tell two
    // similar-sounding threads apart without opening either.
    const parts = [ago(c.created_at)];
    if (c.messages) parts.push(`${c.messages} msg`);
    row.querySelector(".when").textContent = parts.join(" · ");
    row.addEventListener("click", (e) => {
      if (e.target.classList.contains("del")) return;
      openConversation(c.id);
    });
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openConversation(c.id); }
    });
    row.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (state.conversation === c.id) {
        state.conversation = null;
        el.thread.replaceChildren();
      }
      loadConversations();
    });
    el.convs.append(row);
  }
}

async function openConversation(id, { route = true } = {}) {
  state.conversation = id;
  // The gauge describes one conversation's occupancy of the window; the next
  // turn in this one will report its own.
  gauge.used = 0;
  showGauge();
  if (route) {
    const known = state.uuids?.get(id);
    if (known) setRoute(known);
  }
  const messages = await api(`/api/conversations/${id}/messages`);
  el.thread.replaceChildren();
  if (!messages.length) {
    el.thread.innerHTML =
      '<div class="empty" id="empty"><h1>ozgent</h1><p>Say something to begin.</p></div>';
  }
  for (const m of messages) {
    const { body, answer } = addMessage(m.role === "user" ? "user" : "assistant", m.text);
    // A reload must show the whole turn, not just its conclusion.
    imageStrip(body, (m.media ?? []).map((src) => ({ src, alt: "attachment" })));
    if (m.thinking && m.thinking.trim()) {
      const pane = reasoningPane(body);
      pane.innerHTML = markdown(m.thinking);
      dressCode(pane);
    }
    for (const call of m.tool_calls ?? []) {
      const card = openToolCard(answer, call);
      if (call.ok !== undefined) closeToolCard(card, call);
    }
  }
  loadConversations();
  setDrawer(false);
}

/// Create the conversation the next message will be written to.
///
/// Called from `send`, not from the New chat button: a conversation created
/// the moment the button is pressed is a row in the sidebar titled "New chat"
/// that stays that way until something is said, and another one every time
/// the button is pressed again. It is named after the first message, so there
/// is nothing to show before there is one.
async function startConversation() {
  const { id, uuid } = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ model: el.model.value }),
  });
  state.conversation = id;
  setRoute(uuid);
  return id;
}

/// Clear the view and wait for the first message.
function newConversation() {
  state.conversation = null;
  gauge.used = 0;
  showGauge();
  el.thread.replaceChildren();
  el.thread.innerHTML =
    '<div class="empty" id="empty"><h1>ozgent</h1><p>Say something to begin.</p></div>';
  setRoute(null);
  loadConversations();
  setDrawer(false);
  el.input.focus();
}

// ---------------------------------------------------------------- sending

/// Show one image full size.
function openLightbox(src, alt) {
  const box = $("lightbox");
  $("lightbox-img").src = src;
  $("lightbox-img").alt = alt ?? "";
  box.hidden = false;
}

function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightbox-img").src = "";
}

/// Attach a strip of images to a message body.
function imageStrip(body, sources) {
  if (!sources.length) return;
  const strip = document.createElement("div");
  strip.className = "sent-images";
  for (const { src, alt } of sources) {
    const img = document.createElement("img");
    img.src = src;
    img.alt = alt ?? "";
    img.addEventListener("click", () => openLightbox(src, alt));
    strip.append(img);
  }
  body.prepend(strip);
}

/// Read a File as a data URL, which is what the server decodes.
function asDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function send(text) {
  if ((!text.trim() && !state.files.length) || state.streaming) return;
  if (!state.conversation) await startConversation();

  // Only images go to the model: everything else needs a tool to read it, and
  // silently dropping a file the user attached would be worse than saying so.
  const attached = state.files.slice();
  const pictures = attached.filter((f) => f.type.startsWith("image/"));
  const images = await Promise.all(pictures.map(asDataUrl));
  state.files = [];
  renderAttachments();

  addMessage("user", text || pictures.map((f) => f.name).join(", "));
  imageStrip(
    el.thread.lastElementChild.querySelector(".body"),
    pictures.map((f) => ({ src: URL.createObjectURL(f), alt: f.name })),
  );
  const ignored = attached.length - pictures.length;
  if (ignored > 0) {
    el.stat.textContent = `${ignored} non-image attachment(s) ignored`;
  }
  const { body, answer: answerEl } = addMessage("assistant", "");
  answerEl.classList.add("cursor");
  setStreaming(true);
  el.stat.textContent = "loading model...";

  let answer = "";
  let reasoning = "";
  let thinkBox = null;
  // Keyed by call id, not a single "current card". A batch is announced in
  // full before any of it is awaited, so with one variable the first result
  // closed the last card and every other card stayed running for good. A
  // model that issues several calls per round — Ling does, Qwen usually does
  // not — hit that on every turn.
  const toolCards = new Map();
  const controller = new AbortController();
  state.abort = controller;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        conversation: state.conversation,
        model: el.model.value,
        message: text,
        thinking: el.thinking.value,
        tools: state.tools,
        images,
      }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.error ?? res.statusText);
    }

    // Parse the SSE framing by hand; EventSource cannot issue a POST.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const event = JSON.parse(line.slice(5).trim());

        if (event.type === "ready") {
          el.stat.textContent = "generating";
          // The window is fixed when the weights load, so it is known now and
          // does not change again until the model is reloaded.
          gauge.window = event.context;
          showGauge();
        } else if (event.type === "thinking") {
          reasoning += event.text;
          // Only once there is something to show: a model that emits an empty
          // <think></think> must not leave a blank box in the transcript.
          if (reasoning.trim()) {
            thinkBox = thinkBox ?? reasoningPane(body);
            // The pane is collapsed by default, so the summary itself has to
            // show that something is happening.
            thinkBox.closest(".think").classList.add("streaming");
            thinkBox.innerHTML = markdown(reasoning);
            // Follow the reasoning as it arrives, but inside its own pane.
            thinkBox.scrollTop = thinkBox.scrollHeight;
            scrollToTail();
          }
        } else if (event.type === "answer") {
          body.querySelector(".think")?.classList.remove("streaming");
          answer += event.text;
          answerEl.innerHTML = markdown(answer);
          // Re-dressed on every token: the blocks are rebuilt by innerHTML,
          // so the headers have to be put back with them.
          dressCode(answerEl);
          scrollToTail();
        } else if (event.type === "permission") {
          // The turn is stopped on this card until it is answered, so it goes
          // where the eye already is — at the tail, in the flow — rather than
          // in a modal that hides what the model was doing when it asked.
          permissionCard(answerEl, event);
          scrollToTail();
        } else if (event.type === "tool_call") {
          toolCards.set(event.id, openToolCard(answerEl, event));
          scrollToTail();
        } else if (event.type === "tool_result") {
          closeToolCard(toolCards.get(event.id), event);
          toolCards.delete(event.id);
          scrollToTail();
        } else if (event.type === "done") {
          body.querySelector(".think")?.classList.remove("streaming");
          const reused = event.reused ? ` · ${event.reused} reused` : "";
          el.stat.textContent =
            `${event.generated} tok · ${event.tokens_per_second.toFixed(1)}/s${reused}`;
          // What the turn cost: the prompt it processed plus what it wrote.
          gauge.used = (event.prompt ?? 0) + (event.generated ?? 0);
          showGauge();
          // An answer that simply stops looks like a crash. Say why.
          const why = {
            ContextFull: "the context filled up — raise it in Settings > Model",
            TokenLimit: "hit the maximum token limit for this reply",
            Cancelled: "stopped",
          }[event.stop];
          if (why) {
            answerEl.insertAdjacentHTML("beforeend", `<p class="note">${escapeHtml(why)}</p>`);
          }
        } else if (event.type === "error") {
          answerEl.insertAdjacentHTML("beforeend", `<p class="error">${escapeHtml(event.message)}</p>`);
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      answerEl.insertAdjacentHTML("beforeend", `<p class="error">${escapeHtml(e.message)}</p>`);
    }
  } finally {
    // Anything still marked running never got its result: the stream ended
    // first. A card that spins for good is worse than one that says so.
    for (const card of toolCards.values()) {
      card.classList.remove("running");
      card.classList.add("bad");
      const ms = card.querySelector(".ms");
      if (ms) ms.textContent = "no result";
    }
    toolCards.clear();
    body.querySelector(".think")?.classList.remove("streaming");
    answerEl.classList.remove("cursor");
    setStreaming(false);
    state.abort = null;
    loadConversations();
  }
}

// ----------------------------------------------------------- quick panel

let quickState = { model: null, options: null, dirty: false };

/// Build one labelled slider. An untouched control shows the inherited value
/// in grey, so it is always clear whether a number was chosen or defaulted.
function slider(spec, effective, override, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "slider";
  const set = override !== undefined && override !== null;
  const value = set ? Number(override) : Number(effective ?? spec.min);

  wrap.innerHTML =
    '<div class="label"><span></span><span class="value"></span></div>' +
    `<input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}">`;
  wrap.querySelector(".label span").textContent = spec.label;
  const readout = wrap.querySelector(".value");
  const input = wrap.querySelector("input");

  const show = (v, chosen) => {
    readout.textContent =
      spec.zero && Number(v) === 0 ? spec.zero : String(v);
    readout.classList.toggle("inherited", !chosen);
    const pct = ((v - spec.min) / (spec.max - spec.min)) * 100;
    input.style.setProperty("--fill", `${Math.max(0, Math.min(100, pct))}%`);
  };

  input.value = String(value);
  show(value, set);
  input.addEventListener("input", () => {
    show(input.value, true);
    onChange(spec.key, Number(input.value));
  });
  return wrap;
}

function choice(spec, effective, override, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "choice";
  const set = override !== undefined && override !== null;
  const value = String(override ?? effective ?? spec.options[1]);

  wrap.innerHTML =
    '<div class="label"><span></span><span class="value"></span></div>' +
    '<div class="segmented" role="group"></div>';
  wrap.querySelector(".label span").textContent = spec.label;
  const readout = wrap.querySelector(".value");
  readout.textContent = value;
  readout.classList.toggle("inherited", !set);
  wrap.title = spec.hint || "";

  const group = wrap.querySelector(".segmented");
  for (const option of spec.options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = option;
    button.setAttribute("aria-pressed", String(option === value));
    button.addEventListener("click", () => {
      for (const other of group.children) other.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-pressed", "true");
      readout.textContent = option;
      readout.classList.remove("inherited");
      onChange(spec.key, option);
    });
    group.append(button);
  }
  return wrap;
}

async function renderQuick() {
  const model = el.model.value;
  if (!model || el.model.disabled) {
    el.quick.innerHTML = '<p class="hint">Install a model to change its settings.</p>';
    return;
  }
  const data = await api(`/api/models/${encodeURIComponent(model)}/options`);
  quickState = { model, options: data, dirty: false };

  el.quick.replaceChildren();
  const pending = {};
  for (const spec of QUICK) {
    // The ceiling for output tokens is the model's own context: asking for
    // more than the window holds is not a setting, it is a mistake.
    const bounded = bound(spec, data.limits);
    el.quick.append(
      slider(bounded, data.effective[spec.key], data.overrides[spec.key], (key, value) => {
        pending[key] = value;
        quickState.dirty = true;
      }),
    );
  }

  for (const spec of CHOICES) {
    el.quick.append(
      choice(spec, data.effective[spec.key], data.overrides[spec.key], (key, value) => {
        pending[key] = value;
        quickState.dirty = true;
      }),
    );
  }

  const foot = document.createElement("div");
  foot.className = "foot";
  foot.innerHTML =
    '<span class="note"></span><span class="spacer"></span>' +
    '<button type="button" class="ghost-btn auto" id="quick-reset">Reset</button>' +
    '<button type="button" class="primary-btn" id="quick-apply">Apply</button>';
  foot.querySelector(".note").textContent = `${data.model} - saved to config.toml`;
  el.quick.append(foot);

  foot.querySelector("#quick-apply").addEventListener("click", async () => {
    const merged = { ...data.overrides, ...pending };
    await api(`/api/models/${encodeURIComponent(model)}/options`, {
      method: "PUT",
      body: JSON.stringify(merged),
    });
    // True now: load-time settings are compared against what the resident
    // model was loaded with, and a difference forces a reload.
    const reloads = ["context_length", "gpu_layers", "cache_type_k", "cache_type_v", "cpu_moe"];
    el.stat.textContent = reloads.some((k) => k in pending)
      ? "saved - the model reloads on your next message"
      : "saved - applies from your next message";
    renderQuick();
  });
  foot.querySelector("#quick-reset").addEventListener("click", async () => {
    await api(`/api/models/${encodeURIComponent(model)}/options`, {
      method: "PUT",
      body: JSON.stringify({}),
    });
    renderQuick();
  });
}

// ------------------------------------------------------------------ routes

/// Reflect the open conversation in the address bar.
///
/// A chat is a place: it should survive a reload and be shareable as a link,
/// which a single `/` cannot do.
function setRoute(uuid, { replace = false } = {}) {
  const url = uuid ? `/chat?cid=${uuid}` : "/new";
  const how = replace ? "replaceState" : "pushState";
  history[how]({ uuid: uuid ?? null }, "", url);
}

async function openRoute() {
  const params = new URLSearchParams(location.search);
  const cid = params.get("cid");
  if (location.pathname === "/chat" && cid) {
    try {
      const found = await api(`/api/conversations/by-uuid/${encodeURIComponent(cid)}`);
      await openConversation(found.id, { route: false });
      return;
    } catch {
      el.stat.textContent = "that conversation no longer exists";
    }
  }
  // `/new` and anything unrecognised land on an empty chat.
  state.conversation = null;
  el.thread.innerHTML =
    '<div class="empty" id="empty"><h1>ozgent</h1><p>Local models, your machine. Pick a model and start typing.</p></div>';
  loadConversations();
}

// --------------------------------------------------------------- settings

// Every knob the engine exposes, with the input type to render it as. The
// order is the order a person reasons about them: sampling first, then the
// memory/VRAM decisions, then the escape hatches.
const PARAMS = [
  { key: "temperature",    label: "Temperature",     type: "number", step: "0.05" },
  { key: "top_p",          label: "Top-p",           type: "number", step: "0.01" },
  { key: "top_k",          label: "Top-k",           type: "number", step: "1" },
  { key: "min_p",          label: "Min-p",           type: "number", step: "0.01" },
  { key: "repeat_penalty", label: "Repeat penalty",  type: "number", step: "0.05" },
  { key: "repeat_last_n",  label: "Repeat window",   type: "number", step: "1" },
  { key: "max_tokens",     label: "Max tokens",      type: "number", step: "1" },
  { key: "context_length", label: "Context length",  type: "number", step: "256" },
  { key: "gpu_layers",     label: "GPU layers",      type: "text",   placeholder: "auto, off, or a number" },
  { key: "cpu_moe",        label: "CPU MoE layers",  type: "text",   placeholder: "auto, off, all, or a number" },
  { key: "cache_type_k",   label: "KV cache",        type: "select", options: ["", "auto", "f16", "q8_0", "q5_1", "q4_0"] },
  { key: "thinking",       label: "Reasoning",       type: "select", options: ["", "auto", "on", "off"] },
  { key: "flash_attention",label: "Flash attention", type: "select", options: ["", "true", "false"] },
  { key: "tools",          label: "Tools",           type: "select", options: ["", "true", "false"] },
];

let settingsCache = { config: null, options: null, tools: null };

function showTab(name) {
  for (const t of document.querySelectorAll(".tab")) {
    t.setAttribute("aria-selected", String(t.dataset.tab === name));
  }
  for (const p of document.querySelectorAll(".panel")) {
    p.hidden = p.dataset.panel !== name;
  }
}

// ---- model parameters ----

function renderParams(data) {
  const box = $("params");
  box.replaceChildren();
  $("params-for").textContent = `Settings for ${data.model}. Values shown grey are inherited.`;

  for (const spec of PARAMS) {
    const wrap = document.createElement("div");
    wrap.className = "param";
    const override = data.overrides[spec.key];
    const effective = data.effective[spec.key];
    if (override !== undefined && override !== null) wrap.classList.add("set");

    const id = `p-${spec.key}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.innerHTML =
      `<span>${spec.label}</span><span class="inherited">${escapeHtml(String(effective ?? ""))}</span>`;
    wrap.append(label);

    let field;
    if (spec.type === "select") {
      field = document.createElement("select");
      for (const o of spec.options) {
        const opt = document.createElement("option");
        opt.value = o;
        opt.textContent = o === "" ? "inherit" : o;
        field.append(opt);
      }
      field.value = override === undefined || override === null ? "" : String(override);
    } else {
      field = document.createElement("input");
      field.type = spec.type;
      if (spec.step) field.step = spec.step;
      field.placeholder = spec.placeholder ?? "inherit";
      field.value = override === undefined || override === null ? "" : String(override);
    }
    field.id = id;
    field.dataset.key = spec.key;
    field.addEventListener("input", () => wrap.classList.toggle("set", field.value !== ""));
    wrap.append(field);
    box.append(wrap);
  }
}

// Only non-empty fields are sent: an empty box means "inherit", which is a
// removed key rather than a stored null.
function collectParams() {
  const out = {};
  for (const field of $("params").querySelectorAll("[data-key]")) {
    const raw = field.value.trim();
    if (!raw) continue;
    const spec = PARAMS.find((p) => p.key === field.dataset.key);
    if (raw === "true" || raw === "false") out[spec.key] = raw === "true";
    else if (spec.type === "number") {
      const n = Number(raw);
      if (!Number.isNaN(n)) out[spec.key] = n;
    } else out[spec.key] = raw;
  }
  // K and V are one control; llama.cpp wants them set together.
  if (out.cache_type_k) out.cache_type_v = out.cache_type_k;
  return out;
}

// ---- tools ----

function renderTools(data) {
  $("set-tools").checked = data.enabled;

  const select = $("search-provider");
  select.replaceChildren();
  for (const p of data.search_providers) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.needs_key
      ? `${p.name}${p.configured ? " (key set)" : " (needs key)"}`
      : `${p.name} (no key needed)`;
    select.append(opt);
  }
  select.value = data.search_provider;
  syncKeyRow();

  const list = $("tool-list");
  list.replaceChildren();
  if (!data.available.length) {
    list.innerHTML = '<p class="hint">No tools loaded. Check the Python interpreter in config.toml.</p>';
  }
  for (const t of data.available) {
    const row = document.createElement("div");
    row.className = "tool";
    row.innerHTML =
      `<input type="checkbox" data-tool="${escapeHtml(t.name)}"${t.enabled ? " checked" : ""}>` +
      `<div class="meta"><div class="n"></div><div class="d"></div></div>`;
    row.querySelector(".n").textContent = t.name;
    row.querySelector(".d").textContent = t.description.split("\n")[0];
    list.append(row);
  }
}

function syncKeyRow() {
  const chosen = settingsCache.tools?.search_providers.find(
    (p) => p.name === $("search-provider").value,
  );
  $("key-row").hidden = !chosen?.needs_key;
  $("key-hint").textContent = chosen?.needs_key
    ? (chosen.configured
        ? "A key is stored. Leave blank to keep it, or type a new one to replace it."
        : `${chosen.name} needs an API key before it can be used.`)
    : "";
  $("search-key").value = "";
}

// ---- memory ----

async function renderFacts() {
  const list = $("fact-list");
  if (!state.conversation) {
    list.innerHTML = '<p class="hint">Open a conversation to see its memory.</p>';
    return;
  }
  const facts = await api(`/api/conversations/${state.conversation}/facts`);
  list.replaceChildren();
  if (!facts.length) {
    list.innerHTML = '<p class="hint">Nothing remembered yet.</p>';
    return;
  }
  for (const f of facts) {
    const row = document.createElement("div");
    row.className = "fact";
    row.innerHTML =
      `<button class="pin" aria-pressed="${f.pinned}" title="Always keep in context">${f.pinned ? "★" : "☆"}</button>` +
      '<span class="t"></span><button class="del" title="Forget" aria-label="Forget this fact">' +
      '<svg class="ic"><use href="#i-trash"></use></svg></button>';
    row.querySelector(".t").textContent = f.text;
    row.querySelector(".pin").addEventListener("click", async () => {
      await api(`/api/facts/${f.id}`, { method: "PATCH", body: JSON.stringify({ pinned: !f.pinned }) });
      renderFacts();
    });
    row.querySelector(".del").addEventListener("click", async () => {
      await api(`/api/facts/${f.id}`, { method: "DELETE" });
      renderFacts();
    });
    list.append(row);
  }
}

async function previewRecall(query) {
  const out = $("recall-out");
  if (!state.conversation) {
    out.innerHTML = '<p class="hint">Open a conversation first.</p>';
    return;
  }
  const r = await api(`/api/conversations/${state.conversation}/recall`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
  const group = (label, items) =>
    `<div class="group"><div class="k">${label}</div>` +
    (items.length
      ? `<ul>${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`
      : '<div class="none">nothing</div>') +
    `</div>`;
  out.innerHTML =
    group("always in context", r.pinned) +
    group("recalled for this question", r.retrieved.map((h) => (h.seq ? `(message ${h.seq}) ` : "") + h.text)) +
    `<div class="group"><div class="k">window</div>` +
    `<div>${r.recent} recent messages, ${r.elided} older left out, ~${r.tokens_used} tokens</div></div>`;
}

/// Fill the settings page's default-model picker.
///
/// Its options are the composer's, so the two can never disagree about what is
/// installed. "None" is a real choice: it is how a user goes back to picking a
/// model deliberately each time.
function renderDefaultModel(current) {
  const select = $("set-default-model");
  select.replaceChildren();

  const none = document.createElement("option");
  none.value = "";
  none.textContent = "None — choose each time";
  select.append(none);

  for (const option of el.model.options) {
    if (!option.value) continue;
    const copy = document.createElement("option");
    copy.value = option.value;
    copy.textContent = option.textContent;
    select.append(copy);
  }
  // A default naming a model that has since been deleted must not silently
  // become a different one, so it is only applied when it still matches.
  select.value = [...select.options].some((o) => o.value === current) ? current : "";
}

// ---- permissions ----

const RULES = [
  ["allow", "Run it"],
  ["ask", "Ask me"],
  ["deny", "Never"],
];

/// Fill one rule dropdown.
function ruleSelect(select, value) {
  select.replaceChildren();
  for (const [rule, label] of RULES) {
    const option = document.createElement("option");
    option.value = rule;
    option.textContent = label;
    select.append(option);
  }
  select.value = value;
}

/// Draw the permissions page from the server's view of the policy.
///
/// The per-tool rows show the rule in force whether or not it was set by name,
/// because "what will happen if this tool is called" is the question being
/// asked. An inherited rule is marked as such, so clearing an override is a
/// visible change rather than a no-op.
function renderPermissions(view) {
  for (const [key, value] of Object.entries(view.defaults)) {
    const select = document.querySelector(`[data-perm="${key}"]`);
    if (select) ruleSelect(select, value);
  }

  const list = $("perm-list");
  list.replaceChildren();
  if (!view.tools.length) {
    list.innerHTML = '<p class="hint">No tools are loaded.</p>';
  }
  for (const tool of view.tools) {
    const row = document.createElement("div");
    row.className = "perm-row";
    row.innerHTML =
      '<div class="perm-id">' +
        `<span class="nm">${escapeHtml(tool.name)}</span>` +
        `<span class="eff eff-${tool.effect}">${escapeHtml(tool.effect)}</span>` +
        `<span class="desc">${escapeHtml(tool.description)}</span>` +
      '</div>' +
      '<select class="select"></select>' +
      '<button type="button" class="ghost-btn" hidden>Clear</button>';

    const select = row.querySelector("select");
    ruleSelect(select, tool.rule);
    const clear = row.querySelector("button");
    clear.hidden = !tool.overridden;
    clear.title = `Go back to the rule for tools that ${tool.effect}`;

    select.addEventListener("change", async () => {
      await api("/api/permissions", {
        method: "PUT",
        body: JSON.stringify({ tools: { [tool.name]: select.value } }),
      });
      await loadPermissions();
    });
    clear.addEventListener("click", async () => {
      // null, not the current value: clearing has to restore inheritance, and
      // writing back the same rule would look identical and behave differently.
      await api("/api/permissions", {
        method: "PUT",
        body: JSON.stringify({ tools: { [tool.name]: null } }),
      });
      await loadPermissions();
    });
    list.append(row);
  }

  const granted = view.tools.filter((t) => t.granted).map((t) => t.name);
  const box = $("perm-session");
  box.hidden = granted.length === 0;
  $("perm-session-text").textContent =
    `Allowed until ozgent restarts: ${granted.join(", ")}`;
}

async function loadPermissions() {
  renderPermissions(await api("/api/permissions"));
}

for (const select of document.querySelectorAll("[data-perm]")) {
  select.addEventListener("change", async () => {
    await api("/api/permissions", {
      method: "PUT",
      body: JSON.stringify({ [select.dataset.perm]: select.value }),
    });
    await loadPermissions();
  });
}

$("perm-forget").addEventListener("click", async () => {
  await api("/api/permissions", {
    method: "PUT",
    body: JSON.stringify({ clear_session: true }),
  });
  await loadPermissions();
});

// ---- open / save ----

async function openSettings() {
  const dialog = $("settings");
  showTab("general");

  const [config, tools] = await Promise.all([api("/api/settings"), api("/api/tools")]);
  settingsCache.config = config;
  settingsCache.tools = tools;

  $("set-markdown").checked = config.ui.markdown;
  $("set-thinking").checked = config.ui.show_thinking;
  $("set-date").checked = config.ui.date_awareness;
  renderDefaultModel(config.default_model);
  renderTools(tools);
  // Loaded with the rest rather than when the tab is opened: it is one small
  // request, and a tab that shows an empty list for a moment reads as broken.
  await loadPermissions();

  const model = el.model.value;
  if (model && !el.model.disabled) {
    settingsCache.options = await api(`/api/models/${encodeURIComponent(model)}/options`);
    renderParams(settingsCache.options);
  } else {
    $("params").innerHTML = '<p class="hint">Install a model to edit its parameters.</p>';
  }
  renderFacts();
  $("recall-out").replaceChildren();
  dialog.showModal();
}

async function saveSettings() {
  const note = $("save-note");
  note.textContent = "saving...";
  try {
    const config = settingsCache.config;
    config.ui.markdown = $("set-markdown").checked;
    config.ui.show_thinking = $("set-thinking").checked;
    config.ui.date_awareness = $("set-date").checked;
    // An empty selection means "no default"; the field is optional on the
    // Rust side, so it has to be null rather than "".
    config.default_model = $("set-default-model").value || null;
    await api("/api/settings", { method: "PUT", body: JSON.stringify(config) });

    const disabled = [...$("tool-list").querySelectorAll("[data-tool]")]
      .filter((c) => !c.checked)
      .map((c) => c.dataset.tool);
    const key = $("search-key").value;
    await api("/api/tools", {
      method: "PUT",
      body: JSON.stringify({
        enabled: $("set-tools").checked,
        search_provider: $("search-provider").value,
        api_key: key === "" ? null : key,
        disabled,
      }),
    });

    if (settingsCache.options) {
      await api(`/api/models/${encodeURIComponent(settingsCache.options.model)}/options`, {
        method: "PUT",
        body: JSON.stringify(collectParams()),
      });
    }
    note.textContent = "saved";
    setTimeout(() => { note.textContent = ""; }, 2000);
    settingsCache.tools = await api("/api/tools");
    renderTools(settingsCache.tools);
    // The picker's default flag comes from the server, so it has to be re-read
    // rather than assumed from what was just submitted.
    const chosen = el.model.value;
    await loadModels();
    // Changing the default should not switch the conversation's model out from
    // under someone mid-thread.
    if ([...el.model.options].some((o) => o.value === chosen)) el.model.value = chosen;
  } catch (e) {
    note.textContent = e.message;
  }
}

// ------------------------------------------------------------------ models

/// Fill the model picker, opening on the configured default.
///
/// Called again after settings are saved, because the default can be changed
/// there and the picker would otherwise keep pointing at the old one until the
/// page was reloaded.
async function loadModels() {
  const models = await api("/api/models");
  el.model.replaceChildren();
  el.model.disabled = false;

  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.alias || m.reference;
    opt.textContent = m.alias ? `${m.alias} (${m.reference})` : m.reference;
    // The name the server matched on, so the default survives being written
    // as either an alias or a full reference.
    if (m.is_default) opt.dataset.default = "1";
    el.model.append(opt);
  }
  if (!models.length) {
    const opt = document.createElement("option");
    // An <option> with no value attribute reports its text as its value, and
    // that placeholder text would then be offered as a model to default to.
    opt.value = "";
    opt.textContent = "no models installed";
    el.model.append(opt);
    el.model.disabled = true;
    return models;
  }
  // Without a default the browser picks the first option, which is what
  // happened before and is still the right fallback.
  const preferred = el.model.querySelector("option[data-default]");
  if (preferred) el.model.value = preferred.value;
  return models;
}

// ------------------------------------------------------------------- boot

async function boot() {
  // Sets the button's label; the stamp itself was applied by the inline
  // script in the document head, before anything was painted.
  applyTheme(storedTheme());
  syncTools();
  await loadModels();
  await loadConversations();
  await openRoute();
}

window.addEventListener("popstate", openRoute);

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = el.input.value;
  el.input.value = "";
  el.input.style.height = "auto";
  send(text);
});

el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = `${el.input.scrollHeight}px`;
});

// Aborting the fetch closes the connection; axum drops the SSE stream, the
// relay task's send then fails, and dropping its end of the worker channel is
// what actually halts generation on the GPU. The button is not cosmetic.
el.send.addEventListener("click", (e) => {
  if (!state.streaming) return;
  e.preventDefault();
  state.abort?.abort();
});

/// Put the tools control in step with the state it reports.
///
/// Three cues, because one is not enough to read at a glance: the hue, the
/// label's own words, and a globe that is struck through when off.
function syncTools() {
  const button = $("tools-toggle");
  button.setAttribute("aria-pressed", String(state.tools));
  button.querySelector(".pill-label").textContent = state.tools ? "Tools on" : "Tools off";
  setIcon(button.querySelector(".ic"), state.tools ? "i-globe" : "i-globe-off");
  // `data-tip` is what the page draws; `title` waits a second and renders in
  // the desktop's own style, and setting it here left the visible tip stale.
  button.dataset.tip = state.tools
    ? "Tools on: the model may search the web and read files"
    : "Tools off: the model answers from what it knows";
}

$("tools-toggle").addEventListener("click", () => {
  state.tools = !state.tools;
  syncTools();
});

$("lightbox").addEventListener("click", (e) => {
  // Clicking the picture itself should not dismiss it.
  if (e.target.id !== "lightbox-img") closeLightbox();
});
$("lightbox-close").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("lightbox").hidden) closeLightbox();
  else if (el.sidebar.classList.contains("open")) setDrawer(false);
});

$("quick-toggle").addEventListener("click", async () => {
  const open = el.quick.hidden;
  el.quick.hidden = !open;
  $("quick-toggle").setAttribute("aria-expanded", String(open));
  if (open) await renderQuick();
});

$("attach").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => {
  addFiles(e.target.files);
  e.target.value = "";
});

// Paste: only take files, so pasting text still behaves as text.
el.input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files ?? [])];
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
});

// Drag and drop anywhere over the composer.
const veil = $("drop-veil");
let dragDepth = 0;
const composer = $("composer");
composer.addEventListener("dragenter", (e) => {
  e.preventDefault();
  if (++dragDepth === 1) veil.hidden = false;
});
composer.addEventListener("dragover", (e) => e.preventDefault());
composer.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; veil.hidden = true; }
});
composer.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  veil.hidden = true;
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});
el.thread.addEventListener("scroll", syncToLatest, { passive: true });
$("to-latest").addEventListener("click", () => {
  el.thread.scrollTo({ top: el.thread.scrollHeight, behavior: "smooth" });
});
$("to-latest").addEventListener("transitionend", (e) => {
  if (e.propertyName === "opacity" && !e.currentTarget.classList.contains("show")) {
    e.currentTarget.hidden = true;
  }
});

$("theme-toggle").addEventListener("click", () => {
  const at = THEMES.findIndex((t) => t.id === storedTheme());
  applyTheme(THEMES[(at + 1) % THEMES.length].id);
});

$("new-chat").addEventListener("click", newConversation);
$("open-settings").addEventListener("click", openSettings);
/// Open or close the drawer, keeping the scrim and the button in step.
///
/// The drawer used to be a class on the sidebar and nothing else: it covered
/// the thread with no way back except the same small button, which on a phone
/// is behind the drawer that is covering it.
function setDrawer(open) {
  el.sidebar.classList.toggle("open", open);
  $("toggle-sidebar").setAttribute("aria-expanded", String(open));
  let scrim = $("scrim");
  if (open && !scrim) {
    scrim = document.createElement("button");
    scrim.id = "scrim";
    scrim.className = "scrim";
    scrim.setAttribute("aria-label", "Close the sidebar");
    scrim.addEventListener("click", () => setDrawer(false));
    document.body.append(scrim);
    // Next frame, so the transition has a value to animate from.
    requestAnimationFrame(() => scrim.classList.add("show"));
  } else if (!open && scrim) {
    scrim.classList.remove("show");
    scrim.addEventListener("transitionend", () => scrim.remove(), { once: true });
    // A browser that skips the transition never fires the event.
    setTimeout(() => scrim.remove(), 400);
  }
}

$("toggle-sidebar").addEventListener("click", () =>
  setDrawer(!el.sidebar.classList.contains("open")),
);

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
}
$("close-settings").addEventListener("click", () => $("settings").close());
$("cancel-settings").addEventListener("click", () => $("settings").close());
$("save-settings").addEventListener("click", saveSettings);
$("search-provider").addEventListener("change", syncKeyRow);

$("fact-add").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("fact-text").value.trim();
  if (!text || !state.conversation) return;
  await api(`/api/conversations/${state.conversation}/facts`, {
    method: "POST",
    body: JSON.stringify({ text, pinned: true }),
  });
  $("fact-text").value = "";
  renderFacts();
});

$("recall-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("recall-query").value.trim();
  if (q) previewRecall(q);
});

// Reloading parameters when the model changes keeps the panel honest: the
// values shown always belong to the model that would actually answer.
el.model.addEventListener("change", async () => {
  // The quick panel too, and for a sharper reason: it stays open across a
  // model change, so it would go on showing the previous model's numbers
  // while Apply wrote them to that model — the one no longer selected.
  if (!el.quick.hidden && !el.model.disabled) await renderQuick();
  if (!$("settings").open || el.model.disabled) return;
  settingsCache.options = await api(`/api/models/${encodeURIComponent(el.model.value)}/options`);
  renderParams(settingsCache.options);
});

boot().catch((e) => {
  el.thread.innerHTML =
    `<div class="empty"><h1>ozgent</h1><p class="error">${escapeHtml(e.message)}</p></div>`;
});
