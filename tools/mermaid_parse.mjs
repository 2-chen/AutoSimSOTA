// Parse a Mermaid diagram with Mermaid's own parser, and say whether it would render.
//
// Usage:
//     npm install --prefix /somewhere mermaid jsdom
//     node tools/mermaid_parse.mjs "$(cat diagram.txt)"
//     MERMAID_MODULES=/somewhere/node_modules node tools/mermaid_parse.mjs --file RUN.md
//
// Why this exists as a file rather than as a test: Mermaid needs a DOM, so checking a diagram
// for real means a node dependency tree that has no business in this repository's test run.
// The tests assert the invariants that keep a label from breaking the parse; this asserts the
// thing itself, and it is what the invariants were derived from. A diagram that fails to parse
// renders as an error box, so getting this wrong is visible -- but a diagram that parses and
// says something false is not, which is why the renderers are written to invent no edges.
//
// `--file` pulls every ```mermaid block out of a markdown document and parses each.

import { readFileSync } from "node:fs";

const modules = process.env.MERMAID_MODULES
  ? `file://${process.env.MERMAID_MODULES}/mermaid/dist/mermaid.esm.mjs`
  : "mermaid";
const jsdom_url = process.env.MERMAID_MODULES
  ? `file://${process.env.MERMAID_MODULES}/jsdom/lib/api.js`
  : "jsdom";

const { JSDOM } = await import(jsdom_url);
const dom = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", {
  value: dom.window.navigator, configurable: true,
});

const mermaid = (await import(modules)).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

function blocks(text) {
  const out = [];
  const pattern = /```mermaid\n([\s\S]*?)```/g;
  let match;
  while ((match = pattern.exec(text)) !== null) out.push(match[1]);
  return out;
}

const argv = process.argv.slice(2);
let diagrams;
if (argv[0] === "--file") {
  diagrams = blocks(readFileSync(argv[1], "utf8"));
} else {
  diagrams = [argv.join(" ")];
}

let failed = 0;
for (const [index, diagram] of diagrams.entries()) {
  try {
    await mermaid.parse(diagram);
    console.log(`PARSE OK ${index}`);
  } catch (error) {
    failed += 1;
    const message = String(error.message || error).split("\n").slice(0, 8).join(" | ");
    console.log(`PARSE FAIL ${index}: ${message}`);
  }
}
if (!diagrams.length) console.log("no mermaid blocks found");
process.exit(failed ? 1 : 0);
