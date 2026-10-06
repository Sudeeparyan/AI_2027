// Shared technical schematics for editable PowerPoint, SVG and PNG.
"use strict";
const fs = require("fs"), path = require("path"), sharp = require("sharp");
const WIDTH = 1440, HEIGHT = 740;
const P = { ink: "1E1B4B", blue: "4F46E5", fill: "F5F3FF", teal: "0F766E", text: "1F2937" };
function wrap(text, limit = 27) {
  const lines = []; let line = "";
  for (const word of String(text || "").split(/\s+/)) {
    if (line && (line + " " + word).length > limit) { lines.push(line); line = word; }
    else line += (line ? " " : "") + word;
  }
  if (line) lines.push(line);
  return lines;
}
function geometry(d) {
  const nodes = d.steps.map((step, i) => ({ ...step, i, x: 42 + (i < 3 ? i : 5 - i) * 494,
    y: i < 3 ? 42 : 370, w: 368, h: 210 }));
  const segments = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i], b = nodes[i + 1], right = b.x > a.x;
    segments.push(i === 2 ? { x1: a.x + a.w / 2, y1: a.y + a.h + 85, x2: b.x + b.w / 2, y2: b.y - 8 } :
      { x1: right ? a.x + a.w + 8 : a.x - 8, y1: a.y + 95, x2: right ? b.x - 8 : b.x + b.w + 8, y2: b.y + 95 });
  }
  return { nodes, segments };
}
const esc = t => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function svgText(text, x, y, size, color, bold, limit) {
  return `<text font-family="Calibri,Arial,sans-serif" font-size="${size}" fill="#${color}" font-weight="${bold ? 700 : 400}" text-anchor="middle">` +
    wrap(text, limit).map((line, i) => `<tspan x="${x}" y="${y + i * size * 1.13}">${esc(line)}</tspan>`).join("") + "</text>";
}
function svg(d) {
  const { nodes, segments } = geometry(d);
  let out = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}"><rect width="100%" height="100%" fill="white"/><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="#${P.blue}"/></marker></defs>`;
  for (const s of segments) out += `<line x1="${s.x1}" y1="${s.y1}" x2="${s.x2}" y2="${s.y2}" stroke="#${P.blue}" stroke-width="3" marker-end="url(#arrow)"/>`;
  for (const a of nodes) {
    out += `<rect x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" rx="5" fill="#${P.fill}" stroke="#${P.blue}" stroke-width="2"/>`;
    out += svgText(`${a.i + 1}. ${a.title}`, a.x + a.w / 2, a.y + 33, 29, P.ink, true, 25);
    out += svgText(a.detail, a.x + a.w / 2, a.y + 109, 25, P.text, false, 29);
    if (a.code) out += svgText(a.code, a.x + a.w / 2, a.y + 191, Math.min(21, 620 / a.code.length), P.teal, false, 80);
    if (a.i < d.arrows.length) out += svgText("Pass: " + d.arrows[a.i], a.x + a.w / 2, a.y + 241, 24, P.teal, false, 32);
  }
  if (d.feedback) {
    const a = nodes[nodes.length - 1], b = nodes[d.feedback_to || 0];
    const returnY = b.i < 3 ? 14 : 335;
    out += `<path d="M${a.x + a.w / 2},${a.y + a.h + 76} V677 H18 V${returnY} H${b.x + b.w / 2} V${b.y - 5}" fill="none" stroke="#${P.teal}" stroke-width="2" stroke-dasharray="8 5" marker-end="url(#arrow)"/>`;
    out += svgText("Repeat: " + d.feedback, WIDTH / 2, 708, 24, P.teal, false, 98);
  }
  return out + "</svg>";
}
function native(slide, d, addText, box) {
  // Slide text needs a taller block than SVG text. Use inches explicitly so
  // each title, explanation and code name has its own reserved area.
  const sx = box.w / WIDTH, x = px => box.x + px * sx;
  const nodes = d.steps.map((step, i) => ({ ...step, i, x: x(42 + (i < 3 ? i : 5 - i) * 494),
    y: box.y + (i < 3 ? 0 : 2.00), w: 368 * sx, h: 1.47 }));
  const line = (x1, y1, x2, y2, color = P.blue, arrow = false, dash = false) => {
    slide.addShape("line", { x: Math.min(x1, x2), y: Math.min(y1, y2),
      w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
      line: { color, width: 1.6, ...(arrow ? { endArrowType: "triangle" } : {}), ...(dash ? { dashType: "dash" } : {}) } });
  };
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i], b = nodes[i + 1], right = b.x > a.x;
    if (i === 2) line(a.x + a.w / 2, a.y + 1.9, b.x + b.w / 2, b.y - 0.04, P.blue, true);
    else line(right ? a.x + a.w + 0.06 : a.x - 0.06, a.y + 0.75,
      right ? b.x - 0.06 : b.x + b.w + 0.06, b.y + 0.75, P.blue, true);
  }
  for (const a of nodes) {
    slide.addShape("rect", { x: a.x, y: a.y, w: a.w, h: a.h, fill: { color: P.fill }, line: { color: P.blue, width: 1.3 } });
    addText(slide, `${a.i + 1}. ${a.title}`, { x: a.x + 0.08, y: a.y + 0.05, w: a.w - 0.16, h: 0.49, fontSize: 19, color: P.ink, bold: true, align: "center", valign: "middle", margin: 0 });
    addText(slide, a.detail, { x: a.x + 0.10, y: a.y + 0.55, w: a.w - 0.20, h: 0.65, fontSize: 17, color: P.text, align: "center", valign: "middle", margin: 0 });
    if (a.code) addText(slide, a.code, { x: a.x + 0.06, y: a.y + 1.23, w: a.w - 0.12, h: 0.20, fontSize: Math.min(12, (a.w - 0.12) * 72 / (0.60 * a.code.length)), color: P.teal, fontFace: "Consolas", align: "center", valign: "middle", margin: 0 });
    if (a.i < d.arrows.length) addText(slide, "Pass: " + d.arrows[a.i], { x: a.x, y: a.y + 1.51, w: a.w, h: 0.36, fontSize: 15, color: P.teal, align: "center", valign: "middle", margin: 0 });
  }
  if (d.feedback) {
    const a = nodes[nodes.length - 1], b = nodes[d.feedback_to || 0];
    const rail = box.y + 3.99, left = box.x + 0.06, top = box.y + (b.i < 3 ? -0.06 : 1.94);
    line(a.x + a.w / 2, a.y + a.h + 0.40, a.x + a.w / 2, rail, P.teal, false, true);
    line(a.x + a.w / 2, rail, left, rail, P.teal, false, true);
    line(left, rail, left, top, P.teal, false, true);
    line(left, top, b.x + b.w / 2, top, P.teal, false, true);
    line(b.x + b.w / 2, top, b.x + b.w / 2, b.y - 0.02, P.teal, true, true);
    addText(slide, "Repeat: " + d.feedback, { x: box.x + 0.15, y: rail + 0.05, w: box.w - 0.3, h: 0.48, fontSize: 15, color: P.teal, align: "center", valign: "middle", margin: 0 });
  }
}
async function main() {
  const [source, out] = process.argv.slice(2), data = JSON.parse(fs.readFileSync(source, "utf8"));
  fs.mkdirSync(out, { recursive: true });
  for (const d of data.diagrams) {
    const text = svg(d), stem = path.join(out, `beginner_${d.id}`);
    fs.writeFileSync(stem + ".svg", text);
    await sharp(Buffer.from(text)).resize(2160, 1110).png().toFile(stem + ".png");
  }
}
module.exports = { native, geometry, svg, wrap };
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
