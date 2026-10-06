// Regression checks for Markdown conversion defects discovered in visual QA.
// Run after building the packs: node tools/test_teaching_notes.js
"use strict";
const fs = require("fs"), path = require("path"), assert = require("node:assert/strict");
const { marked } = require("marked");
const root = path.resolve(__dirname, "..");
const spec = n => JSON.parse(fs.readFileSync(path.join(root, "build", `week_${String(n).padStart(2, "0")}`, "spec.json"), "utf8"));
const metrics = marked.lexer(spec(3).notes_md).find(t => t.type === "table" && t.rows.some(r => r[0].text.includes("Inception Score")));
assert(metrics, "Missing GAN metrics comparison");
const row = metrics.rows.find(r => r[0].text.includes("Inception Score"));
assert.equal(row.length, 3, "Conditional probability must not create a table column");
assert.match(row[1].text, /p\(y\|\s*x\)/, "The full conditional probability belongs in the Captures cell");
assert.match(row[2].text, /Ignores the real data/, "The limitation must remain in its own cell");
const runtime = marked.lexer(spec(5).notes_md).find(t => t.type === "paragraph" && t.text.includes("Downloads: Tiny Shakespeare"));
assert(runtime, "Missing lab download guidance");
const nested = (tokens, type) => tokens.some(t => t.type === type || (t.tokens && nested(t.tokens, type)));
assert(!nested(runtime.tokens, "del"), "Approximate download sizes must not become strikethrough");
assert(spec(8).notes_md.includes("**Step 4.** Optimise the policy"), "PPO must retain step four after the reward equation");
console.log("Teaching-note regressions: table columns, download sizes and resumed step number passed");
