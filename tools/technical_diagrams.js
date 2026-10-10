// Shared technical schematics for editable PowerPoint, SVG and PNG.
"use strict";
const fs = require("fs"), path = require("path"), sharp = require("sharp");
const WIDTH = 1440, HEIGHT = 820;
const P = { ink: "1E1B4B", blue: "4F46E5", fill: "F5F3FF", teal: "0F766E", text: "1F2937" };
const FONT_WIDTHS = require("./arial_widths.json");
const NBSP = " ";
function fitLines(text,width,size,bold=false) {
  const metrics=FONT_WIDTHS[bold?"bold":"normal"], rows=[];
  const measure=s=>Array.from(s).reduce((sum,c)=>sum+(metrics[c===NBSP?" ":c]??.7)*size,0)*1.035;
  // Keep sizes such as "28 × 28" on one line; only ordinary spaces separate words.
  const glued=String(text||"").replace(/(\d) ([×x]) (\d)/g,`$1${NBSP}$2${NBSP}$3`);
  for(const paragraph of glued.split("\n")) {
    let row="";
    for(const word of paragraph.split(/[ \t]+/).filter(Boolean)) {
      if(row && measure(row+" "+word)>width) {rows.push(row);row=word;}
      else row+=(row?" ":"")+word;
      if(measure(word)>width) throw Error(`Unbreakable label exceeds block: ${word}`);
    }
    if(row) rows.push(row);
  }
  return rows;
}
function wrap(text, limit = 27) {
  const lines = []; let line = "";
  for (const word of String(text || "").split(/\s+/)) {
    if (line && (line + " " + word).length > limit) { lines.push(line); line = word; }
    else line += (line ? " " : "") + word;
  }
  if (line) lines.push(line);
  return lines;
}
function legacyGeometry(d) {
  const nodes = d.steps.map((step, i) => ({ ...step, i, x: 42 + (i < 3 ? i : 5 - i) * 494,
    y: i < 3 ? 42 : 418, w: 368, h: 260 }));
  const segments = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i], b = nodes[i + 1], right = b.x > a.x;
    // Every route begins and ends on a block boundary. The row change uses
    // the right-hand rail, keeping the output label below block 3 readable.
    const points = i === 2
      ? [[a.x + a.w, a.y + 130], [WIDTH - 15, a.y + 130], [WIDTH - 15, b.y + 130], [b.x + b.w, b.y + 130]]
      : [[right ? a.x + a.w : a.x, a.y + 130], [right ? b.x : b.x + b.w, b.y + 130]];
    segments.push({ points, from: i, to: i + 1 });
  }
  let feedback = null;
  if (d.feedback) {
    const a = nodes[nodes.length - 1], b = nodes[d.feedback_to ?? 0];
    const returnY = b.i < 3 ? 14 : 394;
    feedback = { points: [[a.x + a.w / 2, a.y + a.h], [a.x + a.w / 2, 760],
      [18, 760], [18, returnY], [b.x + b.w / 2, returnY], [b.x + b.w / 2, b.y]],
      from: a.i, to: b.i };
  }
  return { nodes, segments, feedback };
}
const esc = t => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function svgText(text, x, y, size, color, bold, limit) {
  return `<text font-family="Calibri,Arial,sans-serif" font-size="${size}" fill="#${color}" font-weight="${bold ? 700 : 400}" text-anchor="middle">` +
    wrap(text, limit).map((line, i) => `<tspan x="${x}" y="${y + i * size * 1.13}">${esc(line)}</tspan>`).join("") + "</text>";
}
function legacySvg(d) {
  const { nodes, segments, feedback } = legacyGeometry(d);
  let out = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" role="img"><title>${esc(d.title)}</title><desc>${esc(d.subtitle)}</desc><rect width="100%" height="100%" fill="white"/><defs>`;
  for (const [name, color] of [["arrow", P.blue], ["repeat", P.teal]])
    out += `<marker id="${name}" markerWidth="9" markerHeight="9" refX="9" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="#${color}"/></marker>`;
  out += "</defs>";
  const route = (s, color, repeat = false) => `<polyline points="${s.points.map(p => p.join(',')).join(' ')}" fill="none" stroke="#${color}" stroke-width="${repeat ? 2 : 3}" ${repeat ? 'stroke-dasharray="8 5"' : ''} marker-end="url(#${repeat ? 'repeat' : 'arrow'})"/>`;
  for (const s of segments) out += route(s, P.blue);
  if (feedback) out += route(feedback, P.teal, true);
  for (const a of nodes) {
    out += `<rect x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" rx="5" fill="#${P.fill}" stroke="#${P.blue}" stroke-width="2"/>`;
    out += svgText(`${a.i + 1}. ${a.title}`, a.x + a.w / 2, a.y + 33, 29, P.ink, true, 25);
    out += svgText(a.detail, a.x + a.w / 2, a.y + 125, 27, P.text, false, 27);
    if (a.code) out += svgText(a.code, a.x + a.w / 2, a.y + 226, 22, P.teal, false, 31);
    if (a.i < d.arrows.length) out += svgText("Output: " + d.arrows[a.i], a.x + a.w / 2, a.y + 291, 24, P.teal, false, 32);
  }
  if (d.feedback) out += svgText("Repeat: " + d.feedback, WIDTH / 2, 790, 24, P.teal, false, 112);
  return out + "</svg>";
}
function legacyNative(slide, d, addText, box) {
  // Use exactly the same routes in PowerPoint and the portable SVG/PNG.
  const sx = box.w / WIDTH, sy = box.h / HEIGHT;
  const x = px => box.x + px * sx, y = px => box.y + px * sy;
  const layout = legacyGeometry(d);
  const nodes = layout.nodes.map(a => ({ ...a, x: x(a.x), y: y(a.y), w: a.w * sx, h: a.h * sy }));
  const line = (x1, y1, x2, y2, color = P.blue, arrow = false, dash = false) => {
    slide.addShape("line", { x: Math.min(x1, x2), y: Math.min(y1, y2),
      w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
      line: { color, width: 1.6, ...(arrow ? { endArrowType: "triangle" } : {}), ...(dash ? { dashType: "dash" } : {}) } });
  };
  const route = (s, color, dash = false) => s.points.slice(1).forEach((p, i) =>
    line(x(s.points[i][0]), y(s.points[i][1]), x(p[0]), y(p[1]), color, i === s.points.length - 2, dash));
  layout.segments.forEach(s => route(s, P.blue));
  if (layout.feedback) route(layout.feedback, P.teal, true);
  for (const a of nodes) {
    slide.addShape("rect", { x: a.x, y: a.y, w: a.w, h: a.h, fill: { color: P.fill }, line: { color: P.blue, width: 1.3 } });
    addText(slide, `${a.i + 1}. ${a.title}`, { x: a.x + 0.08, y: a.y, w: a.w - 0.16, h: 0.60, fontSize: 18, color: P.ink, bold: true, align: "center", valign: "middle", margin: 0 });
    addText(slide, a.detail, { x: a.x + 0.10, y: a.y + 0.60, w: a.w - 0.20, h: a.h - 0.82, fontSize: 17, color: P.text, align: "center", valign: "middle", margin: 0 });
    if (a.code) {
      const code = a.code.length > 32 ? a.code.replace(/\s\/\s/g, " /\n") : a.code;
      addText(slide, code, { x: a.x + 0.06, y: a.y + a.h - 0.21, w: a.w - 0.12, h: 0.20, fontSize: 12, color: P.teal, fontFace: "Consolas", align: "center", valign: "middle", margin: 0 });
    }
    if (a.i < d.arrows.length) addText(slide, "Output: " + d.arrows[a.i], { x: a.x, y: a.y + a.h + 0.05, w: a.w, h: 0.48, fontSize: 15, color: P.teal, align: "center", valign: "middle", margin: 0 });
  }
  if (d.feedback) {
    addText(slide, "Repeat: " + d.feedback, { x: box.x + 0.15, y: y(771), w: box.w - 0.3, h: 49 * sy, fontSize: 14, color: P.teal, align: "center", valign: "middle", margin: 0 });
  }
}
// The pilot layout uses one scene for every renderer. Explicit line breaks,
// text boxes, font sizes and connector points are shared, rather than asking
// PowerPoint and SVG to wrap the same paragraph independently.
const LINEAR_HEIGHT = 530;
const ROLES = {
  data: { color: "0072B2", fill: "E4F2FB", shape: "roundRect", label: "Input data" },
  model: { color: "009E73", fill: "E5F7EF", shape: "rect", label: "Model" },
  loss: { color: "D55E00", fill: "FFF0E6", shape: "roundRect", label: "Loss / update" },
  output: { color: "CC79A7", fill: "F9EAF4", shape: "roundRect", label: "Output" },
  tool: { color: "626B73", fill: "F2F4F5", shape: "rect", label: "Human / tool" },
};
function lines(text, chars) {
  const result = [];
  for (const paragraph of String(text || "").split("\n")) {
    let line = "";
    for (let word of paragraph.split(/\s+/)) {
      // Keep identifiers intact. A split function name cannot be found in code.
      if (line && (line + " " + word).length > chars) { result.push(line); line = word; }
      else line += (line ? " " : "") + word;
    }
    if (line) result.push(line);
  }
  return result;
}
// Vertical bands of the linear scene (pixels): repeat label 8-38 above its
// dashed rail at 44; arrow labels sit directly above the blocks; branch rails
// and their labels run below the blocks; the legend is the last row.
const BOX_Y = 142, BOX_H = 236, LABEL_BOTTOM = 134, REPEAT_RAIL = 44, LEGEND_Y = 492;
const CODE_COLOR = "0F766E";
function linearGeometry(d) {
  const count = d.steps.length, margin = 24, gap = 32;
  const w = (WIDTH - margin * 2 - gap * (count - 1)) / count, mid = BOX_Y + BOX_H / 2;
  const nodes = d.steps.map((s, i) => ({ ...s, i, x: margin + i * (w + gap), y: BOX_Y, w, h: BOX_H,
    role: ({input:"data",human:"tool"}[s.role] || s.role) || (i === 0 ? "data" : i === count - 1 ? "output" : "model") }));
  // A null arrow means "no connector": the next block is an independent input
  // (for example random noise) that receives nothing from its left neighbour.
  const segments = nodes.slice(0, -1).flatMap((a, i) => d.arrows[i] ? [{ from: i, to: i + 1,
    points: [[a.x + a.w, mid], [nodes[i + 1].x, mid]], label: d.arrows[i] }] : []);
  // Bypass branches run under the blocks. Shorter spans take the upper rail;
  // branches sharing a block attach at separate points, ordered so that no
  // vertical segment can cross another branch's rail.
  const raw = (d.branches || []).map((b, i) => ({ ...b, i, span: Math.abs(b.to - b.from) }));
  const railOf = new Map([...raw].sort((a, b) => a.span - b.span || a.i - b.i).map((b, k) => [b.i, k]));
  const ends = new Map();  // block index -> [{branch, other}]
  for (const b of raw) for (const [own, other] of [[b.from, b.to], [b.to, b.from]]) {
    if (!ends.has(own)) ends.set(own, []);
    ends.get(own).push({ b, own, other });
  }
  const attachX = new Map();  // `${branch}:${block}` -> x
  for (const [own, list] of ends) {
    const key = e => e.other < own ? [0, own - e.other] : [1, -(e.other - own)];
    list.sort((p, q) => { const a = key(p), b = key(q); return a[0] - b[0] || a[1] - b[1]; });
    list.forEach((e, j) => attachX.set(`${e.b.i}:${own}`, nodes[own].x + nodes[own].w * (j + 1) / (list.length + 1)));
  }
  const branches = raw.map(b => {
    const rail = BOX_Y + BOX_H + 26 + railOf.get(b.i) * 40, x1 = attachX.get(`${b.i}:${b.from}`), x2 = attachX.get(`${b.i}:${b.to}`);
    return { ...b, points: [[x1, BOX_Y + BOX_H], [x1, rail], [x2, rail], [x2, BOX_Y + BOX_H]], color: ROLES.data.color };
  });
  let feedback = null;
  if (d.feedback) {
    const a = nodes[d.feedback_from ?? nodes.length-1], b = nodes[d.feedback_to ?? 0], rail = REPEAT_RAIL;
    const loopOffset = a.i === b.i ? Math.max(2, Math.min(20, (w + gap - Math.min(210, w + 26)) / 2 - 3)) : 0;
    // A separate dashed rail makes repetition distinct from data movement.
    feedback = { from: a.i, to: b.i, label: d.feedback,
      points: [[a.x + a.w / 2 + loopOffset, a.y],
        [a.x + a.w / 2 + loopOffset, rail],
        [b.x + b.w / 2 - loopOffset, rail],
        [b.x + b.w / 2 - loopOffset, b.y]] };
  }
  return { nodes, segments, branches, feedback, width: WIDTH, height: LINEAR_HEIGHT };
}
function textWidth(content, size, bold = false) {
  return Math.max(...fitLines(content, 100000, size, bold).map(row =>
    Array.from(row).reduce((sum, c) => sum + (FONT_WIDTHS[bold ? "bold" : "normal"][c] ?? .7) * size, 0) * 1.035));
}
function scene(d) {
  const g = linearGeometry(d), items = [];
  function text(content, x, y, w, h, size = 24, color = P.text, bold = false, align = "center") {
    const wrapped = fitLines(content,w,size,bold);
    if (wrapped.length * size * 1.16 > h + 1) throw Error(`${d.id}: text does not fit: ${content}`);
    items.push({ kind: "text", text: wrapped.join("\n"), x, y, w, h, size, color, bold, align });
  }
  const rows = (content, w, size, bold = false) => fitLines(content, w, size, bold).length;
  for (const s of g.segments) {
    items.push({ kind: "route", ...s, color: ROLES.model.color });
    // The label sits directly above its arrow's gap, bottom-aligned on the blocks.
    const a = g.nodes[s.from], b = g.nodes[s.to], centre = (a.x + a.w + b.x) / 2;
    const labelW = Math.min(210, a.w + 26), h = rows(s.label, labelW, 24) * 24 * 1.16;
    if (LABEL_BOTTOM - h < (g.feedback ? REPEAT_RAIL + 4 : 8)) throw Error(`${d.id}: arrow label too long: ${s.label}`);
    text(s.label, centre - labelW/2, LABEL_BOTTOM - h, labelW, h, 24, ROLES.data.color);
  }
  for (const s of g.branches) {
    items.push({ kind: "route", ...s });
    // One line directly under its own rail, only as wide as the words.
    const centre = (s.points[1][0] + s.points[2][0]) / 2, w = textWidth(s.label, 24) + 8;
    if (w > WIDTH - 16) throw Error(`${d.id}: branch label too long: ${s.label}`);
    text(s.label, Math.max(8, Math.min(WIDTH - 8 - w, centre - w / 2)), s.points[1][1] + 4, w, 30, 24, s.color);
  }
  if (g.feedback) {
    items.push({ kind: "route", ...g.feedback, color: ROLES.loss.color, dashed: true });
    // The repeat label sits on top of its own dashed rail.
    const w = textWidth(d.feedback, 24) + 8, centre = (g.feedback.points[1][0] + g.feedback.points[2][0]) / 2;
    if (w > WIDTH - 48) throw Error(`${d.id}: repeat label too long: ${d.feedback}`);
    text(d.feedback, Math.max(24, Math.min(WIDTH - 24 - w, centre - w / 2)), 8, w, 30, 24, "A64200");
  }
  for (const a of g.nodes) {
    const role = ROLES[a.role];
    if (!role) throw Error(`${d.id}: unknown role ${a.role}`);
    items.push({ kind: "block", ...a, ...role });
    // Title, plain-language detail and lab code name stack with clear gaps,
    // centred as one group, so no text touches another band or the outline.
    const parts = [[a.title, a.w - 16, 26, P.ink, true], [a.detail, a.w - 16, 24, P.text, false]];
    if (a.code) parts.push([a.code, a.w - 6, 24, CODE_COLOR, false]);
    const heights = parts.map(([t, w, size, , bold]) => rows(t, w, size, bold) * size * 1.16);
    const total = heights.reduce((s, h) => s + h, 0) + 10 + (a.code ? 8 : 0);
    if (total > a.h - 16) throw Error(`${d.id}: block ${a.i + 1} text is too long: ${a.title}`);
    let y = a.y + (a.h - total) / 2;
    parts.forEach(([t, w, size, color, bold], k) => {
      text(t, a.x + (a.w - w) / 2, y, w, heights[k], size, color, bold);
      y += heights[k] + (k === 0 ? 10 : 8);
    });
  }
  // Legend: colour key for the block roles this diagram uses, then its line styles.
  let x = 24;
  const used = new Set(g.nodes.map(n => n.role));
  for (const r of Object.entries(ROLES).filter(([key]) => used.has(key)).map(([, r]) => r)) {
    items.push({kind: "block", x, y: LEGEND_Y + 3, w: 26, h: 26, ...r});
    const w = textWidth(r.label, 24) + 4;
    text(r.label, x + 34, LEGEND_Y, w, 32, 24, P.text, false, "left");
    x += 34 + w + 26;
  }
  const lines = [["data flow", false, ROLES.model.color]].concat(g.feedback ? [["feedback / repeat", true, ROLES.loss.color]] : []);
  for (const [label, dashed, color] of lines) {
    items.push({kind: "route", legend: true, points: [[x, LEGEND_Y + 16], [x + 44, LEGEND_Y + 16]], color, dashed});
    const w = textWidth(label, 24) + 4;
    text(label, x + 52, LEGEND_Y, w, 32, 24, P.text, false, "left");
    x += 52 + w + 26;
  }
  if (x - 26 > WIDTH - 24) throw Error(`${d.id}: legend is wider than the scene`);
  return { ...g, items };
}
function sceneSvg(d) {
  return svgScene(scene(d), d.title, d.subtitle);
}
function flattenedScene(g) {
  const items=g.items.flatMap(i=>{
    if(i.kind!=="text" || !i.text.includes("\n")) return [i];
    const rows=i.text.split("\n"),lh=i.size*1.16,top=i.y+(i.h-rows.length*lh)/2;
    return rows.map((row,n)=>({...i,text:row,y:top+n*lh,h:lh}));
  });
  return {...g,items};
}
function svgScene(g, title = "Technical diagram", subtitle = "") {
  g=flattenedScene(g);
  let result = `<svg xmlns="http://www.w3.org/2000/svg" width="${g.width}" height="${g.height}" role="img"><title>${esc(title)}</title><desc>${esc(subtitle)} Solid arrows carry data. Dashed arrows show repetition.</desc><rect width="100%" height="100%" fill="white"/><defs>`;
  for (const c of [...new Set(g.items.filter(i => i.kind === "route").map(i => i.color))])
    result += `<marker id="arrow${c}" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#${c}"/></marker>`;
  result += "</defs>";
  for (const i of g.items) {
    if (i.kind === "route") result += `<polyline points="${i.points.map(p => p.join(",")).join(" ")}" fill="none" stroke="#${i.color}" stroke-width="3" ${i.dashed ? 'stroke-dasharray="8 5"' : ''} ${i.arrow === false ? '' : `marker-end="url(#arrow${i.color})"`}/>`;
    else if (i.kind === "block") result += i.shape === "ellipse"
      ? `<ellipse cx="${i.x + i.w/2}" cy="${i.y + i.h/2}" rx="${i.w/2}" ry="${i.h/2}" fill="#${i.fill}" stroke="#${i.color}" stroke-width="2"/>`
      : `<rect x="${i.x}" y="${i.y}" width="${i.w}" height="${i.h}" rx="${i.shape === "roundRect" ? 10 : 0}" fill="#${i.fill}" stroke="#${i.color}" stroke-width="2"/>`;
    else {
      if (i.background) result += `<rect x="${i.x}" y="${i.y}" width="${i.w}" height="${i.h}" fill="#${i.background}"/>`;
      const rows = i.text.split("\n"), lh = i.size * 1.16;
      const first = i.y + (i.h - rows.length * lh) / 2 + i.size * .89;
      const align = i.align || "center", anchor = {left:"start",center:"middle",right:"end"}[align];
      const tx = i.x + (align === "left" ? 0 : align === "right" ? i.w : i.w / 2);
      result += `<text font-family="Arial,sans-serif" font-size="${i.size}" fill="#${i.color}" font-weight="${i.bold ? 700 : 400}" text-anchor="${anchor}">`;
      result += rows.map((row, n) => `<tspan x="${tx}" y="${first + n * lh}">${esc(row)}</tspan>`).join("") + "</text>";
    }
  }
  return result + "</svg>";
}
function sceneNative(slide, d, addText, box) {
  return nativeScene(slide, scene(d), addText, box);
}
function nativeScene(slide, g, addText, box) {
  g=flattenedScene(g);
  const sx = box.w / g.width, sy = box.h / g.height;
  const x = v => box.x + v * sx, y = v => box.y + v * sy;
  for (const i of g.items) {
    if (i.kind === "route") i.points.slice(1).forEach((p, n) => {
      const a = i.points[n], x1 = x(a[0]), y1 = y(a[1]), x2 = x(p[0]), y2 = y(p[1]);
      slide.addShape("line", { x: Math.min(x1,x2), y: Math.min(y1,y2), w: Math.abs(x2-x1), h: Math.abs(y2-y1),
        flipH: x2 < x1, flipV: y2 < y1, line: { color: i.color, width: 1.8,
          ...(i.dashed ? {dashType: "dash"} : {}), ...(i.arrow !== false && n === i.points.length - 2 ? {endArrowType: "triangle"} : {}) }});
    });
    else if (i.kind === "block") slide.addShape(i.shape, { x:x(i.x), y:y(i.y), w:i.w*sx, h:i.h*sy,
      ...(i.shape === "roundRect" ? {rectRadius:.08} : {}),
      fill:{color:i.fill}, line:{color:i.color,width:1.2} });
    else addText(slide, i.text, { x:x(i.x), y:y(i.y), w:i.w*sx, h:i.h*sy, fontSize:i.size*sx*72,
      fontFace:"Arial", color:i.color, bold:i.bold, align:i.align || "center", valign:"middle", margin:0,
      ...(i.background ? {fill:{color:i.background}} : {}),
      breakLine:false, paraSpaceAfter:0, lineSpacingMultiple:1.16 });
  }
}
function geometry(d) { return d.layout === "linear" ? linearGeometry(d) : legacyGeometry(d); }
function svg(d) { return d.layout === "linear" ? sceneSvg(d) : legacySvg(d); }
function native(slide, d, addText, box) {
  return d.layout === "linear" ? sceneNative(slide, d, addText, box) : legacyNative(slide, d, addText, box);
}
async function main() {
  if (process.argv[2] === "--scene") {
    const source = process.argv[3], target = process.argv[4], g = JSON.parse(fs.readFileSync(source, "utf8"));
    const xml = svgScene(g, path.basename(source, ".scene.json"));
    fs.writeFileSync(target.replace(/\.png$/, ".svg"), xml);
    await sharp(Buffer.from(xml)).resize({width: Math.round(g.width * 1.5)}).png().toFile(target);
    return;
  }
  const [source, out] = process.argv.slice(2), data = JSON.parse(fs.readFileSync(source, "utf8"));
  fs.mkdirSync(out, { recursive: true });
  for (const d of data.diagrams) {
    const text = svg(d), stem = path.join(out, `beginner_${d.id}`);
    fs.writeFileSync(stem + ".svg", text);
    await sharp(Buffer.from(text)).resize({ width: WIDTH * 1.5 }).png().toFile(stem + ".png");
  }
}
module.exports = { native, nativeScene, geometry, svg, svgScene, scene, flattenedScene, wrap, fitLines, textWidth, ROLES, WIDTH, HEIGHT, LINEAR_HEIGHT };
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
