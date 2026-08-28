// Tests for the client's markdown renderer.
//
// Run by `markdown_renders_what_models_write` in src/api.rs, which skips when
// node is not installed. The renderer is plain source in app.js with no module
// system, so this pulls the function out by evaluating the file in a context
// with the two globals it closes over.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "app.js"), "utf8");

// Only the renderer and the one helper it uses. Everything above touches
// `document` at load time, and everything below it touches the DOM.
const from = source.indexOf("const escapeHtml");
const to = source.indexOf("-------- rendering");
if (from < 0 || to < 0) {
  console.error("could not find the renderer in app.js; the section markers moved");
  process.exit(1);
}
const body = source.slice(from, source.lastIndexOf("\n", to));
const context = vm.createContext({});
vm.runInContext(
  `${body}\nglobalThis.markdown = markdown; globalThis.highlight = highlight;`,
  context,
);
const { markdown } = context;

let failures = 0;
function check(name, actual, predicate, detail = "") {
  const ok = predicate(actual);
  if (!ok) {
    failures++;
    console.error(`FAIL ${name}${detail ? ` — ${detail}` : ""}`);
    console.error(`     got: ${JSON.stringify(actual)}`);
  }
}
const has = (needle) => (out) => out.includes(needle);
const lacks = (needle) => (out) => !out.includes(needle);

// --- fenced code -----------------------------------------------------------
// The bug this file exists for: the block loop trimmed each line, which ate
// the trailing space of the ` 0 ` marker, so the pattern never matched and
// every code block became a bare digit.
check(
  "a fenced block survives",
  markdown("Here:\n\n```sh\necho hello\n```\n\ndone"),
  has("<pre><code"),
);
check(
  "the code itself is present",
  markdown("```sh\necho hello\n```"),
  has("echo hello"),
);
check(
  "no stray marker digit is left behind",
  markdown("```sh\necho hello\n```"),
  (out) => !/<p>\s*\d+\s*<\/p>/.test(out),
  "the block was dropped and replaced by its index",
);
check(
  "the language is recorded",
  markdown("```python\nx = 1\n```"),
  has('class="lang-python"'),
);
check(
  "a fence with no language still renders",
  markdown("```\nplain\n```"),
  has("<pre><code"),
);
check(
  "markup inside code is shown, not run",
  markdown("```html\n<script>alert(1)</script>\n```"),
  lacks("<script>"),
);
check(
  "markdown inside a fence is left alone",
  markdown("```\n# not a heading\n- not a list\n```"),
  (out) => !out.includes("<h3>") && !out.includes("<li>"),
);
check(
  "a shell script keeps its blank lines and indentation",
  markdown("```bash\nif true; then\n  echo a\n\n  echo b\nfi\n```"),
  (out) => out.includes("  echo a") && out.includes("\n\n  echo b"),
);
// Mid-stream the closing fence has not arrived yet.
check(
  "an unterminated fence renders as code while streaming",
  markdown("Here:\n\n```js\nconst a = 1;"),
  (out) => out.includes("<pre") && out.includes("const a = 1;"),
);
check(
  "two blocks in one message both survive",
  markdown("```\nfirst\n```\n\ntext\n\n```\nsecond\n```"),
  (out) => out.includes("first") && out.includes("second"),
);

check(
  "an unterminated fence is marked as such",
  markdown("```py\nx = 1"),
  has('data-unterminated="1"'),
);
check(
  "a properly closed fence is not marked",
  markdown("```py\nx = 1\n```"),
  lacks("data-unterminated"),
);

// --- thematic breaks -------------------------------------------------------
for (const rule of ["---", "***", "___", "- - -", "----------"]) {
  check(`\`${rule}\` is a rule`, markdown(`a\n\n${rule}\n\nb`), has("<hr>"));
}
check(
  "a rule is not left as literal text",
  markdown("a\n\n---\n\nb"),
  lacks("<p>---</p>"),
);
check(
  "a table separator is still a table, not a rule",
  markdown("| a | b |\n| --- | --- |\n| 1 | 2 |"),
  (out) => out.includes("<table>") && !out.includes("<hr>"),
);
check(
  "a bullet is not mistaken for a rule",
  markdown("- one\n- two"),
  (out) => out.includes("<li>one</li>") && !out.includes("<hr>"),
);

// --- inline ----------------------------------------------------------------
check("inline code renders", markdown("use `ls -la` here"), has("<code>ls -la</code>"));
check(
  "emphasis inside inline code is literal",
  markdown("`a*b*c`"),
  has("<code>a*b*c</code>"),
);
check("bold renders", markdown("**bold**"), has("<strong>bold</strong>"));
check("italic renders", markdown("an *italic* word"), has("<em>italic</em>"));
check("strikethrough renders", markdown("~~gone~~"), has("<del>gone</del>"));
check(
  "links open safely",
  markdown("[docs](https://example.com)"),
  (out) => out.includes('rel="noopener noreferrer"') && out.includes('href="https://example.com"'),
);
check(
  "a javascript: url is not turned into a link",
  markdown("[x](javascript:alert(1))"),
  // The text stays visible, which is harmless; what must not exist is an
  // anchor pointing at it.
  (out) => !/<a[^>]+href="javascript:/i.test(out),
);
check("html in prose is escaped", markdown("<img src=x onerror=y>"), lacks("<img"));

// --- blocks ----------------------------------------------------------------
check("headings render", markdown("## Section"), has("<h4>Section</h4>"));
check("blockquotes render", markdown("> quoted"), has("<blockquote>"));
check(
  "nested lists nest",
  markdown("- one\n  - inner\n- two"),
  (out) => (out.match(/<ul>/g) || []).length === 2,
);
check(
  "ordered lists render",
  markdown("1. first\n2. second"),
  (out) => out.includes("<ol>") && out.includes("<li>first</li>"),
);
check("tables render", markdown("| a |\n| --- |\n| 1 |"), has("<table>"));

// --- syntax highlighting ---------------------------------------------------
// The renderer and the highlighter share a file, so the same harness covers
// both. Escaping matters most here: this one writes HTML from source text.
const { highlight } = context;

check(
  "python keywords and strings separate",
  highlight('def f(x):\n    return "hi"', "python"),
  (out) => out.includes('t-kw">def') && out.includes('t-str">&quot;hi&quot;'),
);
check(
  "comments are marked",
  highlight("# a note\nx = 1", "python"),
  has('t-com"># a note'),
);
check(
  "a triple-quoted docstring is one string, not three",
  highlight('"""line one\nline two"""\nx = 1', "python"),
  // If the scanner closed on the first quote, `line` would be outside it.
  (out) => (out.match(/t-str/g) || []).length === 1,
);
check(
  "an apostrophe in a comment does not colour the rest of the file",
  highlight("# it's fine\nreal_code = 1", "python"),
  has("real_code"),
);
check(
  "shell scripts are highlighted",
  highlight('#!/bin/bash\nif true; then\n  echo "hi"\nfi', "bash"),
  (out) => out.includes('t-kw">if') && out.includes('t-typ">echo'),
);
check(
  "aliases resolve",
  highlight("const x = 1;", "js"),
  has('t-kw">const'),
);
check(
  "an unknown language is left alone but still escaped",
  highlight("<script>alert(1)</script>", "brainfuck"),
  (out) => !out.includes("<script>") && out.includes("&lt;script&gt;"),
);
check(
  "markup inside a string is escaped, not emitted",
  highlight('x = "<img onerror=1>"', "python"),
  (out) => !out.includes("<img") && out.includes("&lt;img"),
);
check(
  "an ampersand survives exactly once",
  highlight("a && b", "javascript"),
  (out) => (out.match(/&amp;/g) || []).length === 2 && !out.includes("&amp;amp;"),
);
check(
  "the visible text is unchanged by colouring",
  highlight('def f():\n    return {"a": 1}', "python"),
  // Strip the spans and unescape: what is left must be the original.
  (out) => {
    const text = out.replace(/<[^>]+>/g, "")
      .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
      .replace(/&amp;/g, "&");
    return text === 'def f():\n    return {"a": 1}';
  },
);
check(
  "numbers are found but not inside identifiers",
  highlight("x1 = 42", "python"),
  (out) => out.includes('t-num">42') && !out.includes('t-num">1'),
);
check(
  "a call is marked as one",
  highlight("print(x)", "python"),
  has('t-typ">print'),
);
check(
  "json keys and values are strings",
  highlight('{"a": 1, "b": true}', "json"),
  (out) => out.includes("t-str") && out.includes('t-num">1') && out.includes('t-kw">true'),
);

// --- the consent wording ---------------------------------------------------
// The panel that asks whether a tool may run. Its wording is the part most
// likely to come out wrong and the least likely to be noticed: it appears
// mid-conversation, is read once, and is gone. An earlier version said
// "Let write_file change files or data - config.py?", which is a form field
// read aloud rather than a question.
const verbFrom = source.indexOf("const CONSENT_VERBS");
const verbTo = source.indexOf("/// One value, short enough");
if (verbFrom < 0 || verbTo < 0) {
  console.error("could not find the consent wording in app.js; the markers moved");
  process.exit(1);
}
vm.runInContext(
  'function clip(t, w) { const f = String(t).replace(/\\s*\\n\\s*/g, " x "); ' +
    'return f.length > w ? f.slice(0, w - 1) + "\u2026" : f; }\n' +
    source.slice(verbFrom, verbTo) +
    "\nglobalThis.consentWording = consentWording;",
  context,
);
const { consentWording } = context;
const asks = (name, effect, args, unfinished = false) => {
  const r = consentWording(name, effect, args, unfinished);
  return { q: r.question.replace(/<[^>]+>/g, ""), meta: r.meta.replace(/<[^>]+>/g, "") };
};

check("a write asks to write, naming the file",
  asks("write_file", "write", { path: "config.py" }).q,
  (q) => q === "Write config.py?");
check("a command asks to run it",
  asks("run_command", "execute", { command: "git status" }).q,
  (q) => q === "Run git status?");
check("a search is not a read",
  // Both are `read`, so the effect alone cannot pick the verb: "Read rust
  // async?" is not English.
  asks("web_search", "read", { query: "rust async" }).q,
  (q) => q === "Search for rust async?");
check("a file read asks to read",
  asks("read_file", "read", { path: "src/main.rs" }).q,
  (q) => q === "Read src/main.rs?");
check("a url is fetched",
  asks("fetch_url", "read", { url: "https://example.com" }).q,
  (q) => q === "Fetch https://example.com?");
check("a call with nothing to name falls back to what it may do",
  asks("write_file", "write", {}).q,
  (q) => q === "Allow write_file to change files or data?");
check("the tool is always named somewhere",
  // A verb in the question must not hide which tool is asking.
  asks("write_file", "write", { path: "config.py" }),
  (r) => r.meta.includes("write_file"));
check("the named argument is not repeated underneath",
  asks("write_file", "write", { path: "config.py", mode: "create" }),
  (r) => !r.meta.includes("config.py") && r.meta.includes("mode: create"));
check("an unfinished call says so in words",
  asks("write_file", "write", { path: "a.txt" }, true).meta,
  has("still being generated"));
check("a long path is bounded, so the dom never holds a whole file",
  // The visible truncation is the CSS's job now; this only guards the size
  // of what gets put into the document.
  asks("write_file", "write", { path: "x".repeat(4000) }).q,
  (q) => q.length < 260);
check("the question mark survives a truncated path",
  // It is outside the span that shrinks. A question with its "?" cut off
  // makes the whole panel look broken.
  consentWording("write_file", "write", { path: "x".repeat(400) }, false).question,
  (html) => html.endsWith('<span class="consent-fix">?</span>'));
check("the shrinking part is the name, not the sentence",
  consentWording("write_file", "write", { path: "a.txt" }, false).question,
  (html) => /<span class="consent-name">a\.txt<\/span>/.test(html));
check("markup in an argument is escaped",
  consentWording("write_file", "write", { path: "<img src=x onerror=y>" }, false).question,
  lacks("<img"));
check("markup in a tool name is escaped",
  consentWording("<script>x</script>", "write", {}, false).question,
  lacks("<script>"));

if (failures) {
  console.error(`\n${failures} markdown test(s) failed`);
  process.exit(1);
}
console.log("markdown: all checks passed");
