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
vm.runInContext(`${body}\nglobalThis.markdown = markdown;`, context);
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

if (failures) {
  console.error(`\n${failures} markdown test(s) failed`);
  process.exit(1);
}
console.log("markdown: all checks passed");
