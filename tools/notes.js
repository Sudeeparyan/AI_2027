// Build a week's Teaching Notes (.docx) from build/week_XX/spec.json (Markdown in spec.notes_md).
// Usage: node tools/notes.js <spec.json> <out.docx>
// Markdown support: # headings, paragraphs (**bold**, *italic*, `code`, links), nested lists,
// GFM tables, fenced code, blockquote callouts ("> **Key idea:** ..."), images:
//   ![caption](fig:<figure id>)   figure with caption
//   ![](eq:<hash>)                display equation (pre-rendered by build_week.py)
//   <!-- pagebreak -->            page break
"use strict";

const fs = require("fs");
const { marked } = require("marked");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, ImageRun, ExternalHyperlink, Header, Footer, PageNumber, AlignmentType,
  BorderStyle, LevelFormat, TableLayoutType, VerticalAlign,
} = require("docx");

const C = {
  ink: "1E1B4B", primary: "4F46E5", tint: "EEF0FF", accent: "EA580C", accentTint: "FFF1E6",
  teal: "0F766E", tealTint: "E6F4F1", red: "B91C1C", redTint: "FDECEC", text: "1F2937",
  muted: "6B7280", codeBg: "F3F4F6", border: "D1D5DB",
};
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1247; // A4, ~2.2 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN; // twips
const PX_PER_TWIP = 96 / 1440;
const MAX_IMG_PX = Math.floor(CONTENT_W * PX_PER_TWIP);

const MIMLO_TEXT = {
  1: "Critically explain the core concepts and mathematical principles underlying generative AI, and demonstrate in-depth knowledge of major generative AI model families, including VAEs, GANs, diffusion models and transformer-based LLMs.",
  2: "Analyse and apply multimodal generative AI techniques, including alignment and cross-modal reasoning across text, image, audio and video modalities.",
  3: "Design, implement and fine-tune generative AI models using contemporary frameworks and parameter-efficient methods.",
  4: "Critically evaluate generative AI outputs using quantitative metrics, human-centered assessments and ethical considerations, including fairness, bias and hallucination risks.",
  5: "Deploy generative AI models in real-world scenarios, utilising contemporary frameworks and implement retrieval-augmented or agentic systems.",
};

let spec;
let listInstance = 0;

// ---------- inline ----------
function decode(s) {
  return String(s)
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
}

// Calibri's italic Greek letters use cursive forms (italic θ looks like ϑ), so inside italic runs such as
// inline maths the Greek letters stay upright and match the upright θ, φ and μ used elsewhere.
function plainRuns(rawText, rawStyle) {
  const { math, ...style } = rawStyle;
  // A number stays on the line of its unit ("14 GB", "7 B", "3 min"); code spans and blocks are not changed.
  const text = rawText.replace(/(\d) (?=(?:GB|MB|KB|TB|ms|min|s|B|M|K|T)\b)/g, "$1 ");
  if (math) {
    // Inline maths (<var>): a single Latin letter is an italic variable; digits, operators, Greek letters and
    // words such as log, KL or data stay upright, as in typeset maths.
    return text.split(/((?<![A-Za-z])[A-Za-z][\u0300-\u036F]*(?![A-Za-z]))/).filter((part) => part).map((part) =>
      new TextRun({ text: part, ...style, italics: /^[A-Za-z]/.test(part) && !/^[A-Za-z]{2}/.test(part) }));
  }
  if (!style.italics) return [new TextRun({ text, ...style })];
  return text.split(/([\u0370-\u03FF\u1F00-\u1FFF]+)/).filter((part) => part).map((part) =>
    new TextRun({ text: part, ...style, italics: !/^[\u0370-\u03FF\u1F00-\u1FFF]+$/.test(part) }));
}

function inline(tokens, base = {}) {
  const out = [];
  // <sub>/<sup> from the inline-maths converter become real Word sub/superscript runs; <var> marks inline maths.
  let sub = 0, sup = 0, math = 0;
  for (const t of tokens || []) {
    const style = { ...base, ...(sub ? { subScript: true } : {}), ...(sup ? { superScript: true } : {}),
      ...(math ? { math: true } : {}) };
    switch (t.type) {
      case "text":
        if (t.tokens && t.tokens.length) out.push(...inline(t.tokens, style));
        else out.push(...plainRuns(decode(t.text), style));
        break;
      case "escape":
        out.push(...plainRuns(decode(t.text), style));
        break;
      case "strong":
        out.push(...inline(t.tokens, { ...style, bold: true }));
        break;
      case "em":
        out.push(...inline(t.tokens, { ...style, italics: true }));
        break;
      case "del":
        out.push(...inline(t.tokens, { ...style, strike: true }));
        break;
      case "codespan":
        out.push(new TextRun({ text: decode(t.text), ...style, font: "Consolas", size: 20, color: C.primary }));
        break;
      case "br":
        out.push(new TextRun({ text: "", break: 1 }));
        break;
      case "link":
        out.push(new ExternalHyperlink({
          link: t.href,
          children: [new TextRun({ text: decode(t.text), ...style, color: C.primary, underline: {} })],
        }));
        break;
      case "image":
        out.push(new TextRun({ text: `[${t.text}]`, ...style }));
        break;
      case "html": {
        const tag = String(t.text).trim().toLowerCase();
        if (tag === "<sub>") sub++;
        else if (tag === "</sub>") sub = Math.max(0, sub - 1);
        else if (tag === "<sup>") sup++;
        else if (tag === "</sup>") sup = Math.max(0, sup - 1);
        else if (tag === "<var>") math++;
        else if (tag === "</var>") math = Math.max(0, math - 1);
        break;
      }
      default:
        if (t.text) out.push(...plainRuns(decode(t.text), style));
    }
  }
  return out;
}

// ---------- blocks ----------
function imageBlock(tok) {
  const href = tok.href || "";
  if (href.startsWith("eq:")) {
    const a = spec.assets.eqs[href.slice(3)];
    if (!a) throw new Error(`Missing equation asset ${href}`);
    let w = (a.w / 300) * 96, h = (a.h / 300) * 96;
    const k = Math.min(1, (MAX_IMG_PX - 20) / w);
    return [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(a.path), altText: { name: "Equation", title: "Equation", description: tok.text || "Display equation" }, transformation: { width: Math.round(w * k), height: Math.round(h * k) } })],
    })];
  }
  if (href.startsWith("fig:")) {
    const a = spec.assets.figures[href.slice(4)];
    if (!a) throw new Error(`Missing figure ${href}`);
    const maxW = Math.min(MAX_IMG_PX - 10, 560);
    const w = maxW, h = Math.round((a.h / a.w) * maxW);
    const k = Math.min(1, 520 / h);
    const out = [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(a.path), altText: { name: tok.text || "Figure", title: tok.text || "Technical diagram", description: tok.text || "Technical diagram" }, transformation: { width: Math.round(w * k), height: Math.round(h * k) } })],
    })];
    if (tok.text) out.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: decode(tok.text), italics: true, size: 19, color: C.muted })],
    }));
    return out;
  }
  throw new Error(`Unsupported image href ${href}`);
}

function paragraphBlock(tok) {
  const toks = tok.tokens || [];
  const meaningful = toks.filter((t) => !(t.type === "text" && !t.text.trim()));
  if (meaningful.length === 1 && meaningful[0].type === "image") return imageBlock(meaningful[0]);
  const label = meaningful.length === 1 && meaningful[0].type === "strong";
  // The exit question stays on the page of its answer (Weeks 2 and 10 had ended with a lone "Answer." line).
  const exitQuestion = meaningful[0] && meaningful[0].type === "strong" && /^Exit question\.?$/.test(meaningful[0].text);
  return [new Paragraph({ spacing: { after: 140, line: 288 }, keepLines: true, keepNext: !!tok._keepNext || label || exitQuestion, children: inline(toks) })];
}

function listBlock(tok, level = 0, inst = null) {
  const ordered = tok.ordered;
  if (inst === null) inst = ++listInstance;
  const out = [];
  for (const item of tok.items) {
    let first = true;
    for (const child of item.tokens) {
      if (child.type === "list") {
        out.push(...listBlock(child, level + 1, ++listInstance));
      } else if (child.type === "text" || child.type === "paragraph") {
        const runs = inline(child.tokens || [{ type: "text", text: child.text }]);
        out.push(new Paragraph({
          keepLines: true,
          numbering: first ? { reference: ordered ? "numbers" : "bullets", level, instance: inst } : undefined,
          indent: first ? undefined : { left: 720 * (level + 1) },
          spacing: { after: 80, line: 276 },
          children: runs,
        }));
        first = false;
      } else if (child.type !== "space") {
        out.push(...block(child));
      }
    }
  }
  return out;
}

function cell(children, opts = {}) {
  return new TableCell({
    children,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: spec?.compact_tables ? 40 : 80, bottom: spec?.compact_tables ? 40 : 80, left: 120, right: 120 },
    verticalAlign: opts.valign || VerticalAlign.TOP,
    borders: opts.borders,
  });
}

const noBorders = {
  top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
};

function boxTable(children, fill) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    layout: TableLayoutType.FIXED,
    rows: [new TableRow({ cantSplit: true, children: [cell(children, { width: CONTENT_W, fill, borders: noBorders })] })],
  });
}

function tableBlock(tok) {
  const nCols = tok.header.length;
  const lens = tok.header.map((h, j) => Math.max(h.text.length, ...tok.rows.map((r) => (r[j] ? r[j].text.length : 0))));
  const weights = lens.map((l) => Math.max(6, Math.min(l, 70)));
  const tot = weights.reduce((a, b) => a + b, 0);
  let widths = weights.map((wt) => (wt / tot) * CONTENT_W);
  // Minimum width = longest single word (10 pt bold ≈ 115 twips per character) + cell margins.
  const cellsText = (j) => [tok.header[j].text, ...tok.rows.map((r) => (r[j] ? r[j].text : ""))];
  const minW = tok.header.map((_, j) => Math.max(...cellsText(j).flatMap((t) => t.replace(/[*`]/g, "").split(/\s+/)).map((w) => w.length)) * 115 + 300);
  for (let it = 0; it < 3; it++) {
    const deficit = widths.reduce((a, w, j) => a + Math.max(0, minW[j] - w), 0);
    if (deficit <= 0) break;
    const donors = widths.map((w, j) => Math.max(0, w - minW[j]));
    const pool = donors.reduce((a, b) => a + b, 0) || 1;
    widths = widths.map((w, j) => (w < minW[j] ? minW[j] : w - (donors[j] / pool) * deficit));
  }
  widths = widths.map((w) => Math.floor(w));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: C.border };
  const borders = { top: border, bottom: border, left: border, right: border };
  const header = new TableRow({
    cantSplit: true,
    tableHeader: true,
    children: tok.header.map((h, j) => cell([new Paragraph({ children: inline(h.tokens, { bold: true, color: "FFFFFF", size: 20 }) })], { width: widths[j], fill: C.ink, borders })),
  });
  const rows = tok.rows.map((r, i) => new TableRow({
    cantSplit: true,
    children: r.map((c, j) => cell([new Paragraph({ children: inline(c.tokens, { size: 20, bold: j === 0 ? true : undefined }) })], { width: widths[j], fill: i % 2 ? "FFFFFF" : C.tint, borders })),
  }));
  return [
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows: [header, ...rows] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function codeBlock(tok) {
  const lines = tok.text.replace(/\s+$/, "").split("\n");
  const paras = lines.map((ln) => new Paragraph({
    spacing: { after: 0, line: 252 },
    children: [new TextRun({ text: ln.length ? ln : " ", font: "Consolas", size: 18, color: /^\s*#/.test(ln) ? C.muted : C.text })],
  }));
  return [boxTable(paras, C.codeBg), new Paragraph({ spacing: { after: 120 }, children: [] })];
}

function calloutBlock(tok) {
  const raw = (tok.text || "").trim();
  const m = raw.match(/^\*\*([^*:]+):?\*\*:?/);
  const label = m ? m[1].trim().toLowerCase() : "";
  let fill = C.tint;
  if (/key idea|remember|in one sentence|definition/.test(label)) fill = C.accentTint;
  else if (/tip|ask|teaching|activity|try|discussion|timing/.test(label)) fill = C.tealTint;
  else if (/misconception|mistake|warning|caution|pitfall|risk/.test(label)) fill = C.redTint;
  const inner = [];
  for (const child of tok.tokens) {
    if (child.type === "paragraph") inner.push(new Paragraph({ spacing: { after: 80, line: 276 }, children: inline(child.tokens) }));
    else if (child.type === "list") inner.push(...listBlock(child));
    else if (child.type !== "space") inner.push(...block(child));
  }
  return [boxTable(inner, fill), new Paragraph({ spacing: { after: 140 }, children: [] })];
}

function headingBlock(tok) {
  const levels = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 };
  if (tok.depth <= 3) {
    return [new Paragraph({ heading: levels[tok.depth], children: inline(tok.tokens), keepNext: true, pageBreakBefore: tok.depth === 1 && tok._pb })];
  }
  return [new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true, children: inline(tok.tokens, { bold: true, color: C.ink }) })];
}

let pendingBreak = false;
function block(tok) {
  switch (tok.type) {
    case "heading": {
      const out = headingBlock({ ...tok, _pb: pendingBreak });
      pendingBreak = false;
      return out;
    }
    case "paragraph": return paragraphBlock(tok);
    case "list": return listBlock(tok);
    case "table": return tableBlock(tok);
    case "code": return codeBlock(tok);
    case "blockquote": return calloutBlock(tok);
    case "html":
      if (/pagebreak/.test(tok.text)) pendingBreak = true;
      return [];
    case "hr": return [new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.border, space: 1 } }, children: [] })];
    case "space": return [];
    default:
      return tok.text ? [new Paragraph({ children: [new TextRun(decode(tok.text))] })] : [];
  }
}

// ---------- cover ----------
function courseCover() {
  // Course-level documents (instructor guide, 7.3 rationale): kicker, title, subtitle and a short meta line.
  const k = (t) => new TextRun({ text: t, bold: true, color: C.accent, size: 24, characterSpacing: 40 });
  return [
    new Paragraph({ spacing: { before: 600, after: 120 }, children: [k(spec.kicker)] }),
    new Paragraph({ style: "Title", spacing: { after: 160 }, children: [new TextRun({ text: spec.title, bold: true, size: 56, color: "000000" })] }),
    new Paragraph({ spacing: { after: 360 }, children: [new TextRun({ text: spec.subtitle || "", size: 28, color: C.muted })] }),
    new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: spec.meta || "", size: 22, color: C.text })] }),
  ];
}

function cover() {
  if (spec.kind === "course") return courseCover();
  const k = (t) => new TextRun({ text: t, bold: true, color: C.accent, size: 24, characterSpacing: 40 });
  const out = [
    new Paragraph({ spacing: { before: 600, after: 120 }, children: [k(`WEEK ${String(spec.week).padStart(2, "0")}  ·  TEACHING NOTES`)] }),
    new Paragraph({ style: "Title", spacing: { after: 160 }, children: [new TextRun({ text: spec.topic, bold: true, size: 56, color: "000000" })] }),
    new Paragraph({ spacing: { after: 360 }, children: [new TextRun({ text: spec.subtitle || "", size: 28, color: C.muted })] }),
    new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Generative AI  ·  MSc in Artificial Intelligence  ·  2-hour lecture + 2-hour lab", size: 22, color: C.text })] }),
  ];
  const border = { style: BorderStyle.SINGLE, size: 4, color: C.border };
  const borders = { top: border, bottom: border, left: border, right: border };
  const w1 = 1900, w2 = CONTENT_W - w1;
  const row = (head, paras) => new TableRow({
    children: [
      cell([new Paragraph({ children: [new TextRun({ text: head, bold: true, color: "FFFFFF", size: 20 })] })], { width: w1, fill: C.ink, borders }),
      cell(paras, { width: w2, fill: C.tint, borders }),
    ],
  });
  const p = (t) => new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: t, size: 20 })] });
  out.push(new Paragraph({ spacing: { before: 360, after: 120 }, children: [new TextRun({ text: "Descriptor Section 7.3 (revised) for this week", bold: true, size: 24, color: C.primary })] }));
  out.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [w1, w2], layout: TableLayoutType.FIXED,
    rows: [row("Lecture topic", [p(spec.topic)]), row("Detail", [p(spec.detail)]), row("Tutorials", spec.tutorials.map(p))],
  }));
  out.push(new Paragraph({ spacing: { before: 300, after: 120 }, children: [new TextRun({ text: "Module learning outcomes (MIMLOs) practised this week", bold: true, size: 24, color: C.primary })] }));
  for (const m of spec.mimlos) {
    out.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: `MIMLO ${m}: `, bold: true, size: 20, color: C.ink }), new TextRun({ text: MIMLO_TEXT[m], size: 20 })] }));
  }
  out.push(new Paragraph({ spacing: { before: 300, after: 120 }, children: [new TextRun({ text: "What is in this week's folder", bold: true, size: 24, color: C.primary })] }));
  const files = [
    ["Lecture slides", `${spec.slides.length} slides with speaker notes, vocabulary and technical diagrams.`],
    ["Teaching notes", "Lecture explanations, worked examples, code maps, practice questions and readings."],
    ["Student lab", "Notebook and Python file with TODOs. Embedded and separate diagrams accompany the lab."],
    ["Lab solutions", "Instructor notebook and Python file with completed code and model answers."],
  ];
  out.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [w1 + 600, w2 - 600], layout: TableLayoutType.FIXED,
    rows: files.map(([a, b], i) => new TableRow({ children: [
      cell([new Paragraph({ keepNext: i < files.length - 1, children: [new TextRun({ text: a, bold: true, size: 20 })] })], { width: w1 + 600, fill: i % 2 ? "FFFFFF" : C.tint, borders }),
      cell([new Paragraph({ keepNext: i < files.length - 1, children: [new TextRun({ text: b, size: 20 })] })], { width: w2 - 600, fill: i % 2 ? "FFFFFF" : C.tint, borders }),
    ] })),
  }));
  return out;
}

// ---------- main ----------
async function main() {
  const [specPath, outPath] = process.argv.slice(2);
  spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  // A non-breaking space keeps each practice-question label "(a)"–"(d)" on the same line as its option;
  // Word otherwise ended lines with a bare "(c)".
  const tokens = marked.lexer(spec.notes_md.replace(/(\*\*\([a-d]\)\*\*) /g, "$1 "));
  pendingBreak = true; // first H1 starts after the cover page
  const body = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    const next = tokens.slice(i + 1).find(candidate => candidate.type !== "space");
    const isFigure = next && next.type === "paragraph" && (next.tokens || []).some(token => token.type === "image");
    // Keep a short diagram lead-in with its picture. The heading already
    // keeps with this paragraph, so the whole introduction travels together.
    body.push(...block(t.type === "paragraph" && isFigure && t.text.length <= 220 ? { ...t, _keepNext: true } : t));
  }

  const doc = new Document({
    creator: "Generative AI module team",
    title: spec.kind === "course" ? spec.title : `Week ${spec.week} Teaching Notes: ${spec.topic}`,
    description: "Teaching notes for the MSc Generative AI module",
    styles: {
      default: { document: { run: { font: "Calibri", size: 22, color: C.text } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 56, bold: true, color: "000000", font: "Calibri" }, paragraph: { spacing: { after: 160 } } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 36, bold: true, color: "000000", font: "Calibri" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, color: "000000", font: "Calibri" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "000000", font: "Calibri" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [0, 1, 2].map((lvl) => ({
          level: lvl, format: LevelFormat.BULLET, text: ["●", "○", "▪"][lvl], alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720 * (lvl + 1) - 360, hanging: 300 } }, run: { color: C.primary } },
        })) },
        { reference: "numbers", levels: [0, 1, 2].map((lvl) => ({
          level: lvl, format: [LevelFormat.DECIMAL, LevelFormat.LOWER_LETTER, LevelFormat.LOWER_ROMAN][lvl], text: `%${lvl + 1}.`, alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720 * (lvl + 1) - 360, hanging: 360 } } },
        })) },
      ],
    },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: spec.kind === "course" ? `Generative AI  ·  ${spec.title}` : `Generative AI  ·  Week ${spec.week}: ${spec.topic}  ·  Teaching Notes`, size: 16, color: C.muted })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: C.muted })] })] }) },
      children: [...cover(), ...body],
    }],
  });
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(outPath, buf);
  console.log(`notes: wrote ${outPath}`);
}

main().catch((e) => {
  console.error(e.stack || e.message);
  process.exit(1);
});
