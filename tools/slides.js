// Build a week's lecture deck (native, editable PowerPoint) from build/week_XX/spec.json.
// Usage: node tools/slides.js <spec.json> <out.pptx>
// The spec is produced by tools/build_week.py from curriculum/weeks/week_XX.yaml.
"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const technicalDiagrams = require("./technical_diagrams");

// ---------- Design system ----------
const C = {
  ink: "1E1B4B", // deep indigo: dark slides, titles
  primary: "4F46E5", // indigo: shapes, numbers
  tint: "EEF0FF", // light indigo: cards
  tint2: "F5F3FF",
  accent: "EA580C", // orange: highlights, key ideas
  accentTint: "FFF1E6",
  teal: "0F766E",
  tealTint: "E6F4F1",
  text: "1F2937",
  muted: "6B7280",
  lav: "C7D2FE", // lavender text on dark
  white: "FFFFFF",
  code: "1E1E2E",
  codeText: "E4E4F0",
  codeComment: "9CA3AF",
};
const FONT = "Calibri";
const MONO = "Consolas";
const W = 13.333, H = 7.5;
const MX = 0.6; // horizontal margin
const CONTENT_TOP = 1.62, CONTENT_BOTTOM = 6.9;
const CW = W - 2 * MX;

const MIMLO_SHORT = {
  1: "Explain core concepts, maths and model families",
  2: "Analyse and apply multimodal generative AI",
  3: "Design, implement and fine-tune models",
  4: "Critically evaluate outputs: metrics, humans, ethics",
  5: "Deploy models; build RAG and agentic systems",
};

// ---------- Helpers ----------
const iconCache = new Map();
async function icon(name, color = C.white, size = 256) {
  const key = `${name}|${color}|${size}`;
  if (iconCache.has(key)) return iconCache.get(key);
  const [set, comp] = (name || "fa:FaLightbulb").split(":");
  let Comp;
  try {
    Comp = require(`react-icons/${set}`)[comp];
  } catch (e) {
    Comp = null;
  }
  if (!Comp) {
    console.warn(`  ! unknown icon ${name}, using fa:FaLightbulb`);
    Comp = require("react-icons/fa").FaLightbulb;
  }
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: "#" + color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  const data = "image/png;base64," + buf.toString("base64");
  iconCache.set(key, data);
  return data;
}

// Parse **bold** and `code` inline markup into pptxgenjs text runs.
// <sub>…</sub> and <sup>…</sup> come from $…$ inline maths converted at build time (build_week.slide_math);
// they become real subscript and superscript runs.
function scriptRuns(text, base = {}) {
  const out = [];
  const re = /<(sub|sup)>(.*?)<\/\1>/g;
  let last = 0, m;
  const s = String(text);
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ text: s.slice(last, m.index), options: { ...base } });
    out.push({ text: m[2], options: { ...base, [m[1] === "sub" ? "subscript" : "superscript"]: true } });
    last = m.index + m[0].length;
  }
  if (last < s.length || !out.length) out.push({ text: s.slice(last), options: { ...base } });
  return out;
}

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  const s = String(text);
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(...scriptRuns(s.slice(last, m.index), base));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(...scriptRuns(tok.slice(2, -2), { ...base, bold: true }));
    else if (tok.startsWith("*")) out.push(...scriptRuns(tok.slice(1, -1), { ...base, italic: true }));
    else out.push({ text: tok.slice(1, -1), options: { ...base, fontFace: MONO, color: base.codeColor || C.primary } });
    last = m.index + tok.length;
  }
  if (last < s.length) out.push(...scriptRuns(s.slice(last), base));
  if (!out.length) out.push({ text: "", options: { ...base } });
  return out;
}
const plain = (t) => String(t).replace(/<\/?su[bp]>/g, "").replace(/\*\*|`|\*/g, "");

// Estimate whether paragraphs fit in a box; return the largest font size that fits.
function fitSize(paras, w, h, maxPt, minPt, opts = {}) {
  minPt = Math.max(14, minPt);
  // Average Calibri glyph width is ~0.47 em (bold ~0.5 em); Consolas is 0.55 em.
  const charW = opts.mono ? 0.56 : opts.bold ? 0.5 : 0.47;
  const lineH = opts.lineH || 1.22;
  const padW = opts.padW ?? 0.2, padH = opts.padH ?? 0.1;
  for (let pt = maxPt; pt >= minPt; pt -= 1) {
    const cpl = Math.max(8, Math.floor((w - padW - (opts.indent || 0)) / ((charW * pt) / 72)));
    let lines = 0;
    for (const p of paras) {
      const t = plain(p.text ?? p);
      const segs = t.split("\n");
      for (const sgm of segs) lines += Math.max(1, Math.ceil(sgm.length / cpl));
    }
    const gap = ((opts.paraGap ?? 6) / 72) * paras.length;
    if ((lines * lineH * pt) / 72 + gap <= h - padH) return pt;
  }
  return minPt;
}

function bulletParas(items, pt, color = C.text) {
  // items: string | {text, sub: [..]}
  const paras = [];
  for (const it of items || []) {
    const t = typeof it === "string" ? it : it.text;
    paras.push({ text: t, level: 0 });
    if (it && typeof it === "object" && Array.isArray(it.sub)) for (const s of it.sub) paras.push({ text: s, level: 1 });
  }
  const arr = [];
  paras.forEach((p, i) => {
    const rs = runs(p.text, { fontSize: p.level ? Math.max(14, pt - 2) : pt, color: p.level ? C.muted : color, fontFace: FONT });
    rs[0].options = {
      ...rs[0].options,
      bullet: p.level ? { indent: 18 } : { code: "25CF", indent: 20 },
      indentLevel: p.level,
      paraSpaceAfter: p.level ? 3 : 8,
      breakLine: false,
    };
    rs[rs.length - 1].options.breakLine = i < paras.length - 1;
    arr.push(...rs);
  });
  return { arr, paras };
}

function addText(slide, content, o) {
  const body = typeof content === "string" && /<su[bp]>/.test(content) ? scriptRuns(content) : content;
  slide.addText(body, { isTextBox: true, fontFace: FONT, color: C.text, valign: "top", margin: 4, ...o });
}

function chrome(slide, spec, idx) {
  slide.background = { color: C.white };
  addText(slide, `WEEK ${String(spec.week).padStart(2, "0")}  ·  ${spec.topic.toUpperCase()}`, {
    x: MX, y: 0.28, w: CW, h: 0.32, fontSize: 11, bold: true, color: C.primary, charSpacing: 1, margin: 0,
  });
  addText(slide, "Generative AI  ·  MSc in Artificial Intelligence", {
    x: MX, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: C.muted, margin: 0,
  });
  addText(slide, String(idx + 1), { x: W - MX - 1, y: 7.05, w: 1, h: 0.3, fontSize: 10, color: C.muted, align: "right", margin: 0 });
}

function title(slide, text) {
  const pt = fitSize([text], CW, 0.95, 32, 22, { bold: true });
  addText(slide, text, { x: MX, y: 0.62, w: CW, h: 0.95, fontSize: pt, bold: true, color: C.ink, valign: "middle", margin: 0 });
}

function imgFit(asset, x, y, w, h, align = "center") {
  // Fit an image with known pixel size into a box, preserving aspect ratio.
  const ar = asset.w / asset.h;
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  let ix = x + (w - iw) / 2;
  if (align === "left") ix = x;
  return { path: asset.path, x: ix, y: y + (h - ih) / 2, w: iw, h: ih };
}

function addFigure(slide, asset, x, y, w, h, align = "center") {
  const placement = imgFit(asset, x, y, w, h, align);
  if (asset.scene) technicalDiagrams.nativeScene(slide, asset.scene, addText, placement);
  else slide.addImage(placement);
  return placement;
}

function callout(slide, text, y, h = 0.85, label = "Key idea") {
  slide.addShape("roundRect", { x: MX, y, w: CW, h, fill: { color: C.accentTint }, line: { color: C.accentTint }, rectRadius: 0.12 });
  const pt = fitSize([label + ": " + text], CW - 0.5, h, 20, 13);
  addText(slide, [
    { text: label + ":  ", options: { bold: true, color: C.accent, fontSize: pt } },
    ...runs(text, { color: C.text, fontSize: pt }),
  ], { x: MX + 0.25, y, w: CW - 0.5, h, valign: "middle" });
}

async function iconCircle(slide, name, x, y, d, fill = C.primary, color = C.white) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  const pad = d * 0.24;
  slide.addImage({ data: await icon(name, color), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}

function numberCircle(slide, n, x, y, d, fill = C.primary) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  addText(slide, String(n), { x, y, w: d, h: d, fontSize: Math.max(14, Math.round(d * 30)), bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
}

// ---------- Layouts ----------
const L = {};

L.course_map = async (slide, s, spec) => {
  title(slide, s.title);
  addFigure(slide, spec.assets.figures[s.figure], MX, CONTENT_TOP, CW, CONTENT_BOTTOM-CONTENT_TOP);
};
L.infographic = async (slide, s, spec) => {
  title(slide, s.title);
  addFigure(slide, spec.assets.figures[s.figure], MX, CONTENT_TOP, CW, 4.98);
  addText(slide, "Exit question: " + s.recap.question, {x:MX,y:6.61,w:CW,h:.37,fontSize:16,color:C.ink,margin:0});
};

L.technical = async (slide, s) => {
  title(slide, s.title);
  const subPt = fitSize([s.diagram.subtitle], CW, 0.42, 18, 14, { paraGap: 0, padH: 0.05, padW: 0.05 });
  addText(slide, s.diagram.subtitle, { x: MX, y: CONTENT_TOP - 0.02, w: CW, h: 0.42, fontSize: subPt, color: C.muted, margin: 0 });
  // Keep the scene's own aspect ratio: native text sizes follow its width.
  const g = technicalDiagrams.geometry(s.diagram);
  const box = imgFit({ w: g.width || technicalDiagrams.WIDTH, h: g.height || technicalDiagrams.HEIGHT }, MX, 2.03, CW, 4.42);
  technicalDiagrams.native(slide, s.diagram, addText, box);
  const check = "Check: " + s.diagram.check_question;
  addText(slide, check, { x: MX, y: 6.48, w: CW, h: 0.52, fontSize: fitSize([check], CW, 0.52, 17, 14), color: C.text, margin: 0, valign: "middle" });
};

L.title = async (slide, s, spec) => {
  slide.background = { color: C.ink };
  slide.addShape("roundRect", { x: 0.8, y: 1.15, w: 1.9, h: 0.5, fill: { color: C.accent }, line: { color: C.accent }, rectRadius: 0.1 });
  addText(slide, `WEEK ${String(spec.week).padStart(2, "0")}`, { x: 0.8, y: 1.15, w: 1.9, h: 0.5, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, charSpacing: 2 });
  const hasCover = s.figure && spec.assets.figures[s.figure];
  const tw = hasCover ? 7.4 : 11.5;
  const tpt = fitSize([spec.topic], tw, 2.3, 45, 32, { bold: true, lineH: 1.1 });
  addText(slide, spec.topic, { x: 0.8, y: 1.9, w: tw, h: 2.3, fontSize: tpt, bold: true, color: C.white, valign: "bottom", margin: 0, lineSpacingMultiple: 0.95 });
  addText(slide, s.subtitle || spec.subtitle || "", { x: 0.8, y: 4.35, w: tw, h: 1.2, fontSize: 20, color: C.lav, margin: 0 });
  addText(slide, "MSc in Artificial Intelligence  ·  Generative AI  ·  2-hour lecture", { x: 0.8, y: 6.45, w: 9, h: 0.4, fontSize: 14, color: C.lav, margin: 0 });
  if (hasCover) {
    addFigure(slide, spec.assets.figures[s.figure], 8.55, 0.9, 4.2, 5.7);
  }
};

L.outcomes = async (slide, s, spec) => {
  title(slide, s.title || "By the end of this week you can…");
  const items = s.items || spec.objectives;
  const lw = 8.3, top = CONTENT_TOP + 0.05;
  const avail = CONTENT_BOTTOM - top;
  const tw = lw - 0.75;
  // Rows get height in proportion to their wrapped line count, at the largest size that fits.
  const linesAt = (t, pt) => Math.max(1, Math.ceil(plain(t).length / Math.floor((tw - 0.15) / ((0.47 * pt) / 72))));
  const rowsAt = (pt) => items.map((t) => (linesAt(t, pt) * 1.2 * pt) / 72 + 0.22);
  // Prefer equal rows (evenly spaced numbers) at a readable size; otherwise
  // give each outcome height in proportion to its wrapped lines.
  let pt = 20, heights = null;
  for (let size = 20; size >= 16 && !heights; size--) {
    const tallest = Math.max(...rowsAt(size));
    if (tallest * items.length <= avail) { pt = size; heights = items.map(() => tallest); }
  }
  if (!heights) {
    for (pt = 20; pt > 12; pt--) if (rowsAt(pt).reduce((a, b) => a + b, 0) <= avail) break;
    heights = rowsAt(pt);
  }
  const spare = (avail - heights.reduce((a, b) => a + b, 0)) / items.length;
  let y = top;
  items.forEach((t, i) => {
    const h = heights[i] + Math.max(0, spare);
    numberCircle(slide, i + 1, MX, y + 0.02, 0.45);
    addText(slide, runs(t, { fontSize: pt }), { x: MX + 0.65, y, w: tw, h, valign: "top" });
    y += h;
  });
  const cx = MX + lw + 0.35, cw = CW - lw - 0.35;
  slide.addShape("roundRect", { x: cx, y: top, w: cw, h: CONTENT_BOTTOM - top, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.12 });
  addText(slide, "Module learning outcomes (MIMLOs) practised", { x: cx + 0.25, y: top + 0.15, w: cw - 0.5, h: 0.7, fontSize: 15, bold: true, color: C.ink });
  const ms = spec.mimlos || [];
  const mRow = Math.min(1.3, (CONTENT_BOTTOM - top - 1.0) / Math.max(1, ms.length));
  ms.forEach((m, i) => {
    const y = top + 0.95 + i * mRow;
    slide.addShape("roundRect", { x: cx + 0.25, y, w: 1.25, h: 0.4, fill: { color: C.primary }, line: { color: C.primary }, rectRadius: 0.08 });
    addText(slide, `MIMLO ${m}`, { x: cx + 0.25, y, w: 1.25, h: 0.4, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    addText(slide, MIMLO_SHORT[m], { x: cx + 0.2, y: y + 0.42, w: cw - 0.4, h: mRow - 0.45, fontSize: 14, color: C.text });
  });
};

L.agenda = async (slide, s) => {
  title(slide, s.title || "Today's plan");
  const items = s.items;
  const cols = items.length > 7 ? 2 : 1;
  const per = Math.ceil(items.length / cols);
  const colW = cols === 1 ? 9.5 : (CW - 0.4) / 2;
  const rowH = Math.min(0.85, (CONTENT_BOTTOM - CONTENT_TOP - 0.1) / per);
  items.forEach((it, i) => {
    const col = Math.floor(i / per), row = i % per;
    const x = MX + col * (colW + 0.4), y = CONTENT_TOP + 0.1 + row * rowH;
    slide.addShape("roundRect", { x, y, w: colW, h: rowH - 0.12, fill: { color: i % 2 ? C.tint2 : C.tint }, line: { color: C.tint }, rectRadius: 0.1 });
    numberCircle(slide, i + 1, x + 0.15, y + (rowH - 0.12 - 0.42) / 2, 0.42);
    const t = typeof it === "string" ? it : it.title;
    addText(slide, runs(t, { fontSize: 19, bold: true, color: C.ink }), { x: x + 0.7, y, w: colW - 2.0, h: rowH - 0.12, valign: "middle" });
    if (it.minutes) addText(slide, `${it.minutes} min`, { x: x + colW - 1.3, y, w: 1.15, h: rowH - 0.12, fontSize: 14, color: C.primary, bold: true, align: "right", valign: "middle" });
  });
  if (s.icon && cols === 1) await iconCircle(slide, s.icon, 10.7, 2.6, 1.9, C.tint, C.primary);
};

L.section = async (slide, s) => {
  slide.background = { color: C.ink };
  addText(slide, String(s.number).padStart(2, "0"), { x: 0.8, y: 1.3, w: 3.2, h: 2.0, fontSize: 110, bold: true, color: C.accent, margin: 0 });
  const pt = fitSize([s.title], 11.5, 1.6, 42, 30, { bold: true });
  addText(slide, s.title, { x: 0.8, y: 3.35, w: 11.5, h: 1.6, fontSize: pt, bold: true, color: C.white, margin: 0 });
  if (s.question) addText(slide, s.question, { x: 0.8, y: 5.05, w: 11.5, h: 1.2, fontSize: 22, italic: true, color: C.lav, margin: 0 });
  if (s.icon) await iconCircle(slide, s.icon, 10.6, 1.2, 1.7, C.primary, C.white);
};

L.bullets = async (slide, s, spec) => {
  title(slide, s.title);
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const fig = s.figure && spec.assets.figures[s.figure];
  let bw = CW;
  if (fig) bw = s.figure_wide ? 5.2 : 6.3;
  else if (s.icon) bw = 8.9;
  const { arr, paras } = bulletParas(s.bullets, 20);
  const pt = fitSize(paras, bw, bottom - CONTENT_TOP, 26, 13, { paraGap: 8, indent: 0.3 });
  const { arr: arr2 } = bulletParas(s.bullets, pt);
  addText(slide, arr2, { x: MX, y: CONTENT_TOP, w: bw, h: bottom - CONTENT_TOP });
  if (fig) {
    const fx = MX + bw + 0.3;
    addFigure(slide, fig, fx, CONTENT_TOP, W - MX - fx, bottom - CONTENT_TOP - (s.caption ? 0.4 : 0));
    if (s.caption) addText(slide, s.caption, { x: fx, y: bottom - 0.4, w: W - MX - fx, h: 0.4, fontSize: 14, italic: true, color: C.muted, align: "center" });
  } else if (s.icon) {
    await iconCircle(slide, s.icon, 10.35, CONTENT_TOP + 0.4, 2.3, C.tint, C.primary);
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.cards = async (slide, s) => {
  title(slide, s.title);
  const cards = s.cards;
  const n = cards.length;
  const grid = n === 4 && s.grid !== "row" ? [2, 2] : n > 4 ? [3, Math.ceil(n / 3)] : [n, 1];
  const [cols, rowsN] = grid;
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const gap = 0.3;
  const cw = (CW - gap * (cols - 1)) / cols;
  const ch = (bottom - CONTENT_TOP - gap * (rowsN - 1)) / rowsN;
  const horizontal = rowsN > 1; // compact: icon left of heading
  const textH = horizontal ? ch - 0.95 : ch - 1.75;
  const pt = Math.min(...cards.map((c) => fitSize([c.text], cw - 0.4, textH, 22, 11)));
  for (let i = 0; i < n; i++) {
    const c = cards[i];
    const x = MX + (i % cols) * (cw + gap), y = CONTENT_TOP + Math.floor(i / cols) * (ch + gap);
    slide.addShape("roundRect", { x, y, w: cw, h: ch, fill: { color: i % 2 ? C.tint2 : C.tint }, line: { color: C.tint }, rectRadius: 0.12 });
    if (horizontal) {
      await iconCircle(slide, c.icon || "fa:FaLightbulb", x + 0.2, y + 0.2, 0.6);
      addText(slide, runs(c.head, { fontSize: 21, bold: true, color: C.ink }), { x: x + 0.9, y: y + 0.18, w: cw - 1.05, h: 0.65, valign: "middle" });
      addText(slide, runs(c.text, { fontSize: pt }), { x: x + 0.2, y: y + 0.9, w: cw - 0.4, h: ch - 0.95 });
    } else {
      await iconCircle(slide, c.icon || "fa:FaLightbulb", x + 0.25, y + 0.25, 0.75);
      addText(slide, runs(c.head, { fontSize: 21, bold: true, color: C.ink }), { x: x + 0.2, y: y + 1.1, w: cw - 0.4, h: 0.6, valign: "middle" });
      addText(slide, runs(c.text, { fontSize: pt }), { x: x + 0.2, y: y + 1.7, w: cw - 0.4, h: ch - 1.75 });
    }
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.compare = async (slide, s) => {
  title(slide, s.title);
  const cols = s.columns;
  const n = cols.length;
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const gap = 0.3;
  const cw = (CW - gap * (n - 1)) / n;
  const fills = [C.primary, C.teal, C.accent];
  const tints = [C.tint, C.tealTint, C.accentTint];
  const bodyH = bottom - CONTENT_TOP - 0.75;
  const pt = Math.min(...cols.map((c) => fitSize(bulletParas(c.points, 16).paras, cw - 0.3, bodyH - 0.1, 24, 11, { paraGap: 8, indent: 0.3 })));
  for (let i = 0; i < n; i++) {
    const c = cols[i];
    const x = MX + i * (cw + gap);
    slide.addShape("roundRect", { x, y: CONTENT_TOP, w: cw, h: 0.62, fill: { color: fills[i % 3] }, line: { color: fills[i % 3] }, rectRadius: 0.1 });
    if (c.icon) slide.addImage({ data: await icon(c.icon, C.white), x: x + 0.18, y: CONTENT_TOP + 0.12, w: 0.38, h: 0.38 });
    addText(slide, runs(c.head, { fontSize: 18, bold: true, color: C.white }), { x: x + (c.icon ? 0.65 : 0.2), y: CONTENT_TOP, w: cw - (c.icon ? 0.8 : 0.4), h: 0.62, valign: "middle" });
    slide.addShape("roundRect", { x, y: CONTENT_TOP + 0.75, w: cw, h: bodyH, fill: { color: tints[i % 3] }, line: { color: tints[i % 3] }, rectRadius: 0.1 });
    addText(slide, bulletParas(c.points, pt).arr, { x: x + 0.12, y: CONTENT_TOP + 0.85, w: cw - 0.24, h: bodyH - 0.15 });
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.table = async (slide, s) => {
  title(slide, s.title);
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const nCols = s.header.length;
  const lens = s.header.map((h, j) => Math.max(plain(h).length, ...s.rows.map((r) => plain(r[j] ?? "").length)));
  const weights = s.col_widths || lens.map((l) => Math.max(8, Math.min(l, 60)));
  const tot = weights.reduce((a, b) => a + b, 0);
  let colW = weights.map((wt) => (wt / tot) * CW);
  // Never let a column be narrower than its longest word (bold, at 20 pt), else words break mid-way.
  const longest = s.header.map((h, j) => Math.max(...[h, ...s.rows.map((r) => r[j] ?? "")].flatMap((c) => plain(c).split(/\s+/)).map((w) => w.length)));
  const minW = longest.map((n) => (n * 0.56 * (s.max_font || 20)) / 72 + 0.25);
  for (let it = 0; it < 3; it++) {
    const deficit = colW.reduce((a, w, j) => a + Math.max(0, minW[j] - w), 0);
    if (deficit <= 0) break;
    const donors = colW.map((w, j) => Math.max(0, w - minW[j]));
    const pool = donors.reduce((a, b) => a + b, 0);
    colW = colW.map((w, j) => (w < minW[j] ? minW[j] : w - (donors[j] / pool) * deficit));
  }
  // Font fit: estimate each row height with the widest-wrapping cell. Projected
  // text never goes below 14 pt; a table that cannot fit must be shortened.
  let pt = s.max_font || 20;
  const tableHeight = (size) => {
    const cpl = colW.map((w) => Math.max(6, Math.floor((w - 0.2) / ((0.47 * size) / 72))));
    return [s.header, ...s.rows].reduce((acc, r) => acc + Math.max(...r.map((c, j) => Math.ceil(plain(c ?? "").length / cpl[j]))) * ((1.25 * size) / 72) + 0.16, 0);
  };
  while (pt > 14 && tableHeight(pt) > bottom - CONTENT_TOP) pt--;
  if (tableHeight(pt) > bottom - CONTENT_TOP) throw new Error("Table does not fit at 14 pt; move rows or detail to the Word notes");
  const rows = [
    s.header.map((h) => ({ text: plain(h), options: { bold: true, color: C.white, fill: { color: C.ink }, fontSize: pt, fontFace: FONT, valign: "middle" } })),
    ...s.rows.map((r, i) =>
      r.map((c, j) => ({
        text: runs(c ?? "", { fontSize: pt, fontFace: FONT, color: C.text, bold: j === 0 && s.first_col_bold !== false }),
        options: { fill: { color: i % 2 ? C.white : C.tint }, valign: "middle" },
      }))
    ),
  ];
  slide.addTable(rows, { x: MX, y: CONTENT_TOP, w: CW, colW, fontSize: pt, fontFace: FONT, border: { type: "solid", pt: 0.5, color: "D1D5DB" }, margin: 0.06, autoPage: false });
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.flow = async (slide, s) => {
  title(slide, s.title);
  const steps = s.steps;
  const n = steps.length;
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const arrowW = 0.45;
  const bw = (CW - arrowW * (n - 1)) / n;
  // Reserve room for bullets (estimated at 18pt), give the rest to the boxes.
  let bulletsH = 0;
  if (s.bullets) {
    const { paras } = bulletParas(s.bullets, 18);
    const cpl = Math.floor((CW - 0.5) / ((0.47 * 18) / 72));
    const lines = paras.reduce((a, p) => a + Math.max(1, Math.ceil(plain(p.text).length / cpl)), 0);
    bulletsH = (lines * 1.25 * 18) / 72 + paras.length * 0.12 + 0.2;
  }
  const boxH = Math.max(2.3, Math.min(3.4, bottom - CONTENT_TOP - 0.2 - bulletsH - (s.bullets ? 0.3 : 0)));
  const y = CONTENT_TOP + 0.15;
  const subPt = Math.min(...steps.map((st) => fitSize([st.sub || ""], bw - 0.25, boxH - 1.5, 20, 11)));
  const labPt = Math.min(...steps.map((st) => fitSize([st.label], bw - 0.2, 0.85, 21, 13, { bold: true })));
  steps.forEach((st, i) => {
    const x = MX + i * (bw + arrowW);
    const hl = st.highlight;
    slide.addShape("roundRect", { x, y, w: bw, h: boxH, fill: { color: hl ? C.accentTint : C.tint }, line: { color: hl ? C.accent : C.primary, width: 1.25 }, rectRadius: 0.12 });
    numberCircle(slide, i + 1, x + bw / 2 - 0.22, y + 0.14, 0.44, hl ? C.accent : C.primary);
    addText(slide, runs(st.label, { fontSize: labPt, bold: true, color: C.ink }), { x: x + 0.08, y: y + 0.62, w: bw - 0.16, h: 0.85, align: "center", valign: "middle" });
    if (st.sub) addText(slide, runs(st.sub, { fontSize: subPt, color: C.text }), { x: x + 0.1, y: y + 1.5, w: bw - 0.2, h: boxH - 1.55, align: "center" });
    if (i < n - 1) {
      slide.addShape("rightArrow", { x: x + bw + 0.07, y: y + boxH / 2 - 0.16, w: arrowW - 0.14, h: 0.32, fill: { color: C.muted }, line: { color: C.muted } });
    }
  });
  if (s.bullets) {
    const by = y + boxH + 0.3;
    const { paras } = bulletParas(s.bullets, 18);
    const pt = fitSize(paras, CW, bottom - by, 20, 12, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(s.bullets, pt).arr, { x: MX, y: by, w: CW, h: bottom - by });
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.figure = async (slide, s, spec) => {
  title(slide, s.title);
  const fig = spec.assets.figures[s.figure];
  if (!fig) throw new Error(`Missing figure ${s.figure}`);
  const bottom = (s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM) - (s.caption ? 0.45 : 0);
  if (s.bullets && s.stack) {
    // Wide figure on top, bullets underneath.
    const { paras } = bulletParas(s.bullets, 18);
    const cpl = Math.floor((CW - 0.5) / ((0.47 * 18) / 72));
    const lines = paras.reduce((a, p) => a + Math.max(1, Math.ceil(plain(p.text).length / cpl)), 0);
    const bh = Math.min(2.6, (lines * 1.25 * 18) / 72 + paras.length * 0.12 + 0.15);
    const fh = bottom - CONTENT_TOP - bh - 0.15;
    const im = addFigure(slide, fig, MX, CONTENT_TOP, CW, fh);
    const by = im.y + im.h + 0.2;
    const pt = fitSize(paras, CW, bottom - by, 20, 12, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(s.bullets, pt).arr, { x: MX, y: by, w: CW, h: bottom - by });
    if (s.caption) addText(slide, s.caption, { x: MX, y: bottom + 0.02, w: CW, h: 0.42, fontSize: 14, italic: true, color: C.muted, align: "center" });
  } else if (s.bullets) {
    const fw = s.figure_width || 7.3;
    addFigure(slide, fig, MX, CONTENT_TOP, fw, bottom - CONTENT_TOP);
    const bx = MX + fw + 0.3, bw = W - MX - bx;
    const { paras } = bulletParas(s.bullets, 18);
    const pt = fitSize(paras, bw, bottom - CONTENT_TOP, 20, 12, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(s.bullets, pt).arr, { x: bx, y: CONTENT_TOP, w: bw, h: bottom - CONTENT_TOP });
    if (s.caption) addText(slide, s.caption, { x: MX, y: bottom + 0.02, w: fw, h: 0.42, fontSize: 14, italic: true, color: C.muted, align: "center" });
  } else {
    addFigure(slide, fig, MX, CONTENT_TOP, CW, bottom - CONTENT_TOP);
    if (s.caption) addText(slide, s.caption, { x: MX, y: bottom + 0.02, w: CW, h: 0.42, fontSize: 14, italic: true, color: C.muted, align: "center" });
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.equation = async (slide, s, spec) => {
  title(slide, s.title);
  const eqs = s.eq_assets || [];
  const panelH = s.panel_h || Math.min(2.4, 0.95 * eqs.length + 0.5);
  slide.addShape("roundRect", { x: MX, y: CONTENT_TOP, w: CW, h: panelH, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.12 });
  // Equations are rendered at a fixed point size (300 dpi, 1 in = 300 px). Rows get heights in proportion to each
  // equation's natural height, so a tall line (fractions, sums) is not squeezed and all lines share one type size
  // unless a line is too wide for the panel.
  const gap = 0.1;
  const natHs = eqs.map((a) => a.h / 300);
  const avail = panelH - 0.3 - gap * Math.max(0, eqs.length - 1);
  const kH = Math.min(1.0, avail / Math.max(0.01, natHs.reduce((x, y) => x + y, 0)));
  const used = natHs.reduce((x, y) => x + y * kH, 0) + gap * Math.max(0, eqs.length - 1);
  const pad = Math.max(0, panelH - 0.3 - used) / (eqs.length + 1);  // spread spare height evenly around the lines
  let y = CONTENT_TOP + 0.15 + pad;
  eqs.forEach((a) => {
    const natW = a.w / 300, natH = a.h / 300;
    const rowH = natH * kH;
    const k = Math.min(kH, (CW - 0.6) / natW);
    const iw = natW * k, ih = natH * k;
    slide.addImage({ path: a.path, x: MX + (CW - iw) / 2, y: y + (rowH - ih) / 2, w: iw, h: ih });
    y += rowH + gap + pad;
  });
  const top = CONTENT_TOP + panelH + 0.25;
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const hasTerms = s.terms && s.terms.length;
  const lw = hasTerms ? 6.6 : CW;
  if (s.read_as) {
    addText(slide, "Read it as", { x: MX, y: top, w: lw, h: 0.4, fontSize: 15, bold: true, color: C.primary });
    const readItems = Array.isArray(s.read_as) ? s.read_as : [s.read_as];
    const { paras } = bulletParas(readItems, 17);
    const pt = fitSize(paras, lw, bottom - top - 0.45, 20, 12, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(readItems, pt).arr, { x: MX, y: top + 0.42, w: lw, h: bottom - top - 0.45 });
  }
  if (hasTerms) {
    const tx = MX + lw + 0.3, tw = W - MX - tx;
    addText(slide, "Symbols", { x: tx, y: top, w: tw, h: 0.4, fontSize: 15, bold: true, color: C.primary });
    const rowH = Math.min(0.75, (bottom - top - 0.45) / s.terms.length);
    const pt = Math.min(...s.terms.map(([a, b]) => fitSize([b], tw - 1.35, rowH, 17, 10)));
    s.terms.forEach(([sym, meaning], i) => {
      const y = top + 0.45 + i * rowH;
      addText(slide, sym, { x: tx, y, w: 1.25, h: rowH, fontSize: pt + 1, bold: true, color: C.ink, fontFace: "Cambria Math", valign: "middle" });
      addText(slide, runs(meaning, { fontSize: pt }), { x: tx + 1.3, y, w: tw - 1.3, h: rowH, valign: "middle" });
    });
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.code = async (slide, s) => {
  title(slide, s.title);
  const bottom = s.callout ? CONTENT_BOTTOM - 1.05 : CONTENT_BOTTOM;
  const hasB = s.bullets && s.bullets.length;
  const lines = String(s.code).replace(/\s+$/, "").split("\n");
  const maxLen = Math.max(...lines.map((l) => l.length));
  const CH = 0.55; // Consolas glyph width in em
  // Width left for code: panel inset 0.15 in each side, text margins 6 pt each side, plus a small safety gap.
  const fits = (w, h, pt) => (maxLen * CH * pt) / 72 <= w - 0.6 && (lines.length * 1.2 * pt) / 72 <= h - 0.3;
  // Choose the arrangement that keeps projected code at least 14 pt.
  let side = hasB && fits(7.6, bottom - CONTENT_TOP, 14);
  let cw = side || !hasB ? (hasB ? 7.6 : CW) : CW;
  let ch = bottom - CONTENT_TOP;
  let bh = 0;
  if (hasB && !side) {
    const { paras } = bulletParas(s.bullets, 16);
    const cpl = Math.floor((CW - 0.5) / ((0.47 * 16) / 72));
    const nl = paras.reduce((acc, p) => acc + Math.max(1, Math.ceil(plain(p.text).length / cpl)), 0);
    bh = Math.min(1.9, (nl * 1.25 * 16) / 72 + paras.length * 0.1 + 0.15);
    ch = bottom - CONTENT_TOP - bh - 0.15;
  }
  // Largest size that fits, from 20 pt down to the 14 pt floor.
  let pt = 20;
  while (pt > 14 && !fits(cw, ch, pt)) pt--;
  if (!fits(cw, ch, pt)) throw new Error("Code excerpt does not fit at 14 pt; shorten the slide and keep the full example in Word notes");
  // Shrink the dark panel to the code when it is short (no large empty block); bullets move up with it.
  if (side || !hasB) ch = Math.min(ch, (lines.length * 1.22 * pt) / 72 + 0.55);
  slide.addShape("roundRect", { x: MX, y: CONTENT_TOP, w: cw, h: ch, fill: { color: C.code }, line: { color: C.code }, rectRadius: 0.1 });
  const codeRuns = [];
  lines.forEach((ln, i) => {
    const isComment = /^\s*#/.test(ln);
    codeRuns.push({ text: ln.length ? ln : " ", options: { fontFace: MONO, fontSize: pt, color: isComment ? C.codeComment : C.codeText, breakLine: i < lines.length - 1 } });
  });
  addText(slide, codeRuns, { x: MX + 0.15, y: CONTENT_TOP + 0.12, w: cw - 0.3, h: ch - 0.24, margin: 6, lineSpacingMultiple: 1.0 });
  if (hasB && side) {
    const bx = MX + cw + 0.3, bw = W - MX - bx;
    const { paras } = bulletParas(s.bullets, 17);
    const bpt = fitSize(paras, bw, bottom - CONTENT_TOP, 20, 11, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(s.bullets, bpt).arr, { x: bx, y: CONTENT_TOP, w: bw, h: bottom - CONTENT_TOP });
  } else if (hasB) {
    const by = CONTENT_TOP + ch + 0.15;
    const { paras } = bulletParas(s.bullets, 16);
    const bpt = fitSize(paras, CW, bottom - by, 20, 11, { paraGap: 6, indent: 0.3 });
    addText(slide, bulletParas(s.bullets, bpt).arr, { x: MX, y: by, w: CW, h: bottom - by });
  }
  if (s.callout) callout(slide, s.callout, CONTENT_BOTTOM - 0.9, 0.9, s.callout_label || "Key idea");
};

L.quiz = async (slide, s) => {
  title(slide, s.title || "Quick check");
  await iconCircle(slide, "fa:FaQuestion", MX, CONTENT_TOP + 0.05, 0.7, C.accent);
  const qpt = fitSize([s.question], CW - 1.0, 1.2, 24, 16, { bold: true });
  addText(slide, runs(s.question, { fontSize: qpt, bold: true, color: C.ink }), { x: MX + 0.95, y: CONTENT_TOP, w: CW - 0.95, h: 1.2, valign: "middle" });
  const opts = s.options;
  const cols = 2, gap = 0.3;
  const top = CONTENT_TOP + 1.45;
  const rowsN = Math.ceil(opts.length / cols);
  const ow = (CW - gap) / cols, oh = Math.min(1.45, (CONTENT_BOTTOM - 0.45 - top - gap * (rowsN - 1)) / rowsN);
  const pt = Math.min(...opts.map((o) => fitSize([o], ow - 1.0, oh - 0.1, 20, 12)));
  opts.forEach((o, i) => {
    const x = MX + (i % cols) * (ow + gap), y = top + Math.floor(i / cols) * (oh + gap);
    slide.addShape("roundRect", { x, y, w: ow, h: oh, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.1 });
    numberCircle(slide, "ABCDEF"[i], x + 0.18, y + (oh - 0.5) / 2, 0.5);
    addText(slide, runs(o, { fontSize: pt }), { x: x + 0.85, y, w: ow - 1.0, h: oh, valign: "middle" });
  });
  addText(slide, s.prompt || "Vote, then convince your neighbour (1 minute).", { x: MX, y: CONTENT_BOTTOM - 0.4, w: CW, h: 0.4, fontSize: 14, italic: true, color: C.muted });
};

L.callout = async (slide, s) => {
  slide.background = { color: C.white };
  slide.addShape("roundRect", { x: 0.9, y: 1.1, w: W - 1.8, h: 5.3, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.2 });
  await iconCircle(slide, s.icon || "fa:FaKey", 1.4, 1.6, 0.9, C.accent);
  addText(slide, (s.label || "Key idea").toUpperCase(), { x: 2.5, y: 1.75, w: 8, h: 0.6, fontSize: 16, bold: true, color: C.accent, charSpacing: 2, valign: "middle" });
  const pt = fitSize([s.statement], W - 3.2, 2.6, 34, 22, { bold: true });
  addText(slide, runs(s.statement, { fontSize: pt, bold: true, color: C.ink }), { x: 1.5, y: 2.6, w: W - 3.0, h: 2.6, valign: "middle" });
  if (s.sub) addText(slide, runs(s.sub, { fontSize: 18, color: C.text }), { x: 1.5, y: 5.2, w: W - 3.0, h: 1.0, valign: "top" });
};

L.stat = async (slide, s) => {
  title(slide, s.title);
  const st = s.stats;
  const n = st.length, gap = 0.3;
  const cw = (CW - gap * (n - 1)) / n;
  const bottom = s.caption ? CONTENT_BOTTOM - 0.9 : CONTENT_BOTTOM;
  // A value must stay on one line, and all cards share one size. fitSize assumes at least 8 characters per line,
  // so "≈ 140 GB" passed at 60 pt and wrapped onto two lines above its card. 0.52 em per bold character.
  const vpt = Math.max(30, Math.min(60, ...st.map((x) => Math.floor(((cw - 0.5) * 72) / (0.52 * plain(x.value).length)))));
  st.forEach((x, i) => {
    const X = MX + i * (cw + gap);
    slide.addShape("roundRect", { x: X, y: CONTENT_TOP + 0.2, w: cw, h: bottom - CONTENT_TOP - 0.2, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.12 });
    addText(slide, x.value, { x: X + 0.15, y: CONTENT_TOP + 0.45, w: cw - 0.3, h: 1.5, fontSize: vpt, bold: true, color: C.primary, align: "center", valign: "middle" });
    const lpt = fitSize([x.label], cw - 0.4, bottom - CONTENT_TOP - 2.3, 19, 11);
    addText(slide, runs(x.label, { fontSize: lpt }), { x: X + 0.2, y: CONTENT_TOP + 2.05, w: cw - 0.4, h: bottom - CONTENT_TOP - 2.3, align: "center" });
  });
  if (s.caption) addText(slide, runs(s.caption, { fontSize: 14, color: C.muted, italic: true }), { x: MX, y: bottom + 0.15, w: CW, h: 0.7 });
};

L.lab = async (slide, s) => {
  title(slide, s.title || "This week's lab");
  const lw = 7.9;
  const steps = s.steps;
  const rowH = Math.min(0.95, (CONTENT_BOTTOM - CONTENT_TOP) / steps.length);
  const pt = Math.min(...steps.map((t) => fitSize([t], lw - 0.8, rowH - 0.05, 20, 12)));
  steps.forEach((t, i) => {
    const y = CONTENT_TOP + i * rowH;
    numberCircle(slide, i + 1, MX, y + 0.05, 0.45, C.teal);
    addText(slide, runs(t, { fontSize: pt }), { x: MX + 0.65, y, w: lw - 0.7, h: rowH });
  });
  const cx = MX + lw + 0.3, cw = CW - lw - 0.3;
  slide.addShape("roundRect", { x: cx, y: CONTENT_TOP, w: cw, h: CONTENT_BOTTOM - CONTENT_TOP, fill: { color: C.tealTint }, line: { color: C.tealTint }, rectRadius: 0.12 });
  const facts = [
    ["fa:FaLaptopCode", "Runtime", s.runtime || "Google Colab (free T4 GPU) or local Jupyter"],
    ["fa:FaClock", "Time", s.time || "2-hour lab"],
    ["fa:FaClipboardCheck", "Hand in", s.deliverable || "Completed notebook with answers"],
  ];
  let y = CONTENT_TOP + 0.2;
  const fh = (CONTENT_BOTTOM - CONTENT_TOP - 0.3) / facts.length;
  for (const [ic, head, text] of facts) {
    await iconCircle(slide, ic, cx + 0.2, y + 0.05, 0.55, C.teal);
    addText(slide, head, { x: cx + 0.9, y, w: cw - 1.0, h: 0.4, fontSize: 15, bold: true, color: C.ink });
    const fpt = fitSize([text], cw - 1.05, fh - 0.45, 14, 10);
    addText(slide, runs(text, { fontSize: fpt }), { x: cx + 0.9, y: y + 0.38, w: cw - 1.05, h: fh - 0.45 });
    y += fh;
  }
};

L.summary = async (slide, s) => {
  title(slide, s.title || "Key takeaways");
  const pts = s.points;
  const cols = pts.length > 5 ? 2 : 1;
  const per = Math.ceil(pts.length / cols);
  const colW = (CW - 0.4 * (cols - 1)) / cols;
  const rowH = (CONTENT_BOTTOM - CONTENT_TOP) / per;
  const pt = Math.min(...pts.map((t) => fitSize([t], colW - 0.8, rowH - 0.1, 19, 12)));
  const check = await icon("fa:FaCheckCircle", C.teal);
  pts.forEach((t, i) => {
    const col = Math.floor(i / per), row = i % per;
    const x = MX + col * (colW + 0.4), y = CONTENT_TOP + row * rowH;
    slide.addImage({ data: check, x, y: y + 0.08, w: 0.42, h: 0.42 });
    addText(slide, runs(t, { fontSize: pt }), { x: x + 0.6, y, w: colW - 0.65, h: rowH - 0.05 });
  });
};

L.resources = async (slide, s) => {
  title(slide, s.title || "Read, watch, practise");
  const kinds = { book: "fa:FaBook", paper: "fa:FaFileAlt", doc: "fa:FaGlobe", video: "fa:FaYoutube", course: "fa:FaGraduationCap", tool: "fa:FaTools" };
  const items = s.items;
  const cols = items.length > 6 ? 2 : 1;
  const per = Math.ceil(items.length / cols);
  const colW = (CW - 0.4 * (cols - 1)) / cols;
  const rowH = Math.min(0.9, (CONTENT_BOTTOM - CONTENT_TOP) / per);
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const col = Math.floor(i / per), row = i % per;
    const x = MX + col * (colW + 0.4), y = CONTENT_TOP + row * rowH;
    await iconCircle(slide, kinds[it.kind] || "fa:FaLink", x, y + 0.08, 0.5, it.kind === "video" ? "DC2626" : C.primary);
    const label = [{ text: it.label, options: { fontSize: 15, bold: true, color: C.ink, breakLine: !!it.note } }];
    if (it.note) label.push({ text: it.note, options: { fontSize: 14, color: C.muted } });
    const o = { x: x + 0.65, y, w: colW - 0.7, h: rowH - 0.05, valign: "middle" };
    if (it.url) label[0].options.hyperlink = { url: it.url };
    addText(slide, label, o);
  }
};

L.discussion = async (slide, s) => {
  title(slide, s.title || "Discuss");
  await iconCircle(slide, s.icon || "fa:FaComments", MX, CONTENT_TOP + 0.05, 1.1, C.teal);
  const ppt = fitSize([s.prompt], CW - 1.5, 1.6, 26, 17, { bold: true });
  addText(slide, runs(s.prompt, { fontSize: ppt, bold: true, color: C.ink }), { x: MX + 1.4, y: CONTENT_TOP, w: CW - 1.4, h: 1.6, valign: "middle" });
  if (s.points) {
    const { paras } = bulletParas(s.points, 18);
    const pt = fitSize(paras, CW - 1.4, CONTENT_BOTTOM - CONTENT_TOP - 2.0, 22, 12, { paraGap: 8, indent: 0.3 });
    addText(slide, bulletParas(s.points, pt).arr, { x: MX + 1.4, y: CONTENT_TOP + 1.85, w: CW - 1.4, h: CONTENT_BOTTOM - CONTENT_TOP - 1.9 });
  }
};

// ---------- Main ----------
async function main() {
  const [specPath, outPath] = process.argv.slice(2);
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = `Week ${spec.week}: ${spec.topic}`;
  pres.subject = "Generative AI (MSc in Artificial Intelligence)";
  pres.author = "Generative AI module team";
  for (let i = 0; i < spec.slides.length; i++) {
    const s = spec.slides[i];
    const slide = pres.addSlide();
    const fn = L[s.type];
    if (!fn) throw new Error(`Slide ${i + 1}: unknown type ${s.type}`);
    if (!["title", "section"].includes(s.type)) chrome(slide, spec, i);
    try {
      await fn(slide, s, spec);
    } catch (e) {
      throw new Error(`Slide ${i + 1} (${s.type}: ${s.title || ""}): ${e.message}`);
    }
    if (s.notes) slide.addNotes(String(s.notes).trim());
  }
  await pres.writeFile({ fileName: outPath });
  console.log(`slides: wrote ${outPath} (${spec.slides.length} slides)`);
}

main().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
