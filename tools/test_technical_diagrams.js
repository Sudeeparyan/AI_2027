// Check actual diagram topology and shared rendering geometry, not screenshots.
"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {geometry, svg, native, scene, nativeScene, svgScene, flattenedScene, textWidth, WIDTH, HEIGHT} = require("./technical_diagrams");
// The inked part of a text item: its widest row and its rows' height, placed as the renderers place them.
function ink(t) {
  const rows = t.text.split("\n"), lh = t.size * 1.16, w = Math.max(...rows.map(r => textWidth(r, t.size, t.bold)));
  const x = t.align === "left" ? t.x : t.align === "right" ? t.x + t.w - w : t.x + (t.w - w) / 2;
  return { x, y: t.y + (t.h - rows.length * lh) / 2, w, h: rows.length * lh };
}
function segmentHitsBox([x1, y1], [x2, y2], b) {
  const pad = 1;  // a line may pass 1 px from glyphs, never through them
  const lo = (a, c) => Math.min(a, c), hi = (a, c) => Math.max(a, c);
  return hi(x1, x2) > b.x + pad && lo(x1, x2) < b.x + b.w - pad && hi(y1, y2) > b.y + pad && lo(y1, y2) < b.y + b.h - pad;
}
const sharp=require("sharp"), pilotScenes=[];
const {recapScene,courseMapScene}=require("./teaching_support");
const boundary = (p,a) => ((p[0]===a.x || p[0]===a.x+a.w) && p[1]>=a.y && p[1]<=a.y+a.h)
  || ((p[1]===a.y || p[1]===a.y+a.h) && p[0]>=a.x && p[0]<=a.x+a.w);
function nativeItems(draw) {
  const shapes=[],texts=[];
  draw({addShape:(kind,o)=>shapes.push({kind,...o})}, (_slide,text,o)=>texts.push({text,...o}));
  return {shapes,texts};
}
function crosses(a,b,c,d) {
  // Orthogonal routes only; shared endpoints are valid junctions.
  const between=(v,x,y)=>v>Math.min(x,y)+1e-6 && v<Math.max(x,y)-1e-6;
  if(a[0]===b[0] && c[1]===d[1]) return between(a[0],c[0],d[0]) && between(c[1],a[1],b[1]);
  if(a[1]===b[1] && c[0]===d[0]) return between(c[0],a[0],b[0]) && between(a[1],c[1],d[1]);
  return false;
}
function inspectScene(g,label) {
  for(const i of g.items) {
    if(i.kind==="route") {
      for(const [x,y] of i.points) assert(x>=0 && x<=g.width && y>=0 && y<=g.height,`${label}: connector outside scene`);
    } else {
      assert(i.x>=-.01 && i.y>=-.01 && i.x+i.w<=g.width+.01 && i.y+i.h<=g.height+.01,`${label}: ${i.text || i.kind} outside scene`);
      if(i.kind==="text") {
        assert(i.text.split("\n").length*i.size*1.16<=i.h+.01,`${label}: text exceeds allotted height`);
        assert(i.size*12.133/g.width*72>=14,`${label}: font below 14 pt at full slide width`);
      }
    }
  }
  const rendered=nativeItems((slide,addText)=>nativeScene(slide,g,addText,{x:0,y:0,w:g.width,h:g.height}));
  const expectedTexts=flattenedScene(g).items.filter(i=>i.kind==="text");
  assert.equal(rendered.texts.length,expectedTexts.length);
  for(let j=0;j<expectedTexts.length;j++) {
    const actual=rendered.texts[j], expected=expectedTexts[j];
    for(const key of ["text","x","y","w","h","color","bold"]) assert.equal(actual[key],expected[key],`${label}: portable/native ${key} differs`);
    assert.equal(actual.fontSize,expected.size*72);
  }
  assert.equal(rendered.shapes.filter(i=>i.kind!=="line").length,g.items.filter(i=>i.kind==="block").length);
  const xml=svgScene(g);
  for(const t of expectedTexts) assert(xml.includes(`font-size="${t.size}"`));
  assert.equal((xml.match(/<polyline /g)||[]).length,g.items.filter(i=>i.kind==="route").length);
}
let count=0,linear=0;
for(const filename of fs.readdirSync(path.join(__dirname,"../curriculum/beginner"))) {
  if(!filename.endsWith(".json")) continue;
  const data=JSON.parse(fs.readFileSync(path.join(__dirname,"../curriculum/beginner",filename),"utf8"));
  for(const d of data.diagrams) {
    const g=geometry(d), routes=[...g.segments,...(g.branches||[]),...(g.feedback?[g.feedback]:[])];
    for(const r of routes) {
      assert(boundary(r.points[0],g.nodes[r.from]),`${filename}/${d.id}: disconnected start`);
      assert(boundary(r.points.at(-1),g.nodes[r.to]),`${filename}/${d.id}: disconnected end`);
      assert(r.points.every(([x,y])=>x>=0&&x<=WIDTH&&y>=0&&y<=(g.height||HEIGHT)),"route outside canvas");
    }
    const {shapes}=nativeItems((s,a)=>native(s,d,a,{x:0,y:0,w:g.width||WIDTH,h:g.height||HEIGHT}));
    // Linear scenes add one-segment legend samples: solid data flow, plus dashed repeat when the scene repeats.
    const legendLines=d.layout==="linear"?scene(d).items.filter(i=>i.kind==="route"&&i.legend).length:0;
    if(d.layout==="linear") assert.equal(legendLines,g.feedback?2:1,`${d.id}: line-style legend missing`);
    assert.equal(shapes.filter(s=>s.kind==="line").length,routes.reduce((n,r)=>n+r.points.length-1,0)+legendLines);
    assert.equal(shapes.filter(s=>s.line?.endArrowType).length,routes.length+legendLines,"native arrow missing");
    assert.equal((svg(d).match(/<polyline /g)||[]).length,routes.length+legendLines);
    if(d.layout==="linear") {
      linear++; assert(d.steps.length<=7);
      for(const r of g.segments) {
        assert(r.label,"Data arrow has no label");
        for(let i=1;i<r.points.length;i++) assert(r.points[i][0]>=r.points[i-1][0],"main flow moves backwards");
      }
      for(let i=0;i<routes.length;i++) for(let j=i+1;j<routes.length;j++)
        for(let a=1;a<routes[i].points.length;a++) for(let b=1;b<routes[j].points.length;b++)
          assert(!crosses(routes[i].points[a-1],routes[i].points[a],routes[j].points[b-1],routes[j].points[b]),`${d.id}: arrows cross`);
      const layout=scene(d); inspectScene(layout,`${filename}/${d.id}`); pilotScenes.push({layout,label:`${filename}/${d.id}`});
      // The key lists exactly the block roles drawn, so no entry describes a block that is not there.
      const roleLabels={data:"Input data",model:"Model",loss:"Loss / update",output:"Output",tool:"Human / tool"};
      const used=new Set(g.nodes.map(n=>n.role));
      for(const [role,legend] of Object.entries(roleLabels))
        assert.equal(layout.items.some(i=>i.kind==="text"&&i.text===legend),used.has(role),`${d.id}: legend entry ${legend}`);
      assert(layout.items.some(i=>i.kind==="text"&&i.text==="data flow"),`${d.id}: missing data flow legend`);
      assert.equal(layout.items.some(i=>i.kind==="text"&&i.text==="feedback / repeat"),Boolean(g.feedback),`${d.id}: repeat legend`);
      // No connector passes through any text (routes are horizontal or vertical segments).
      for(const r of layout.items.filter(i=>i.kind==="route"))
        for(let k=1;k<r.points.length;k++)
          for(const t of layout.items.filter(i=>i.kind==="text"))
            assert(!segmentHitsBox(r.points[k-1],r.points[k],ink(t)),`${filename}/${d.id}: a connector crosses text: ${t.text}`);
      // Every text item stays clear of every block outline it does not belong to.
      const blocks=layout.items.filter(i=>i.kind==="block"&&i.w>30);
      for(const t of layout.items.filter(i=>i.kind==="text")) {
        const inside=blocks.filter(b=>t.x>=b.x-.01&&t.x+t.w<=b.x+b.w+.01&&t.y>=b.y-.01&&t.y+t.h<=b.y+b.h+.01);
        for(const b of blocks) if(!inside.includes(b)) {
          const overlap=Math.min(t.x+t.w,b.x+b.w)-Math.max(t.x,b.x)>0&&Math.min(t.y+t.h,b.y+b.h)-Math.max(t.y,b.y)>0;
          assert(!overlap,`${filename}/${d.id}: text crosses a block outline: ${t.text}`);
        }
        for(const [k,other] of layout.items.entries()) if(other.kind==="text"&&other!==t&&layout.items.indexOf(t)<k) {
          const overlap=Math.min(t.x+t.w,other.x+other.w)-Math.max(t.x,other.x)>0&&Math.min(t.y+t.h,other.y+other.h)-Math.max(t.y,other.y)>0;
          assert(!overlap,`${filename}/${d.id}: text overlaps text: ${t.text} / ${other.text}`);
        }
      }
      if(data.week===5 && d.id==="mechanism") assert(g.branches.some(b=>b.from===1&&b.to===5&&b.label.includes("V")),"Values must bypass scores and reach weighted mixture");
      if(data.week===5 && d.id==="training") assert(g.branches.some(b=>b.from===0&&b.to===4&&b.label.includes("targets")),"Targets must bypass model and reach loss");
      if(data.week===5 && d.id==="overview") assert(d.steps.some(s=>s.code==="head")&&d.steps.some(s=>s.code==="softmax"),"Output head and softmax must both be visible");
    } else assert.equal(shapes.filter(s=>s.kind==="rect").length,d.steps.length);
    count++;
  }
  for(const [id,layout] of [["course_map",courseMapScene(data.week)],...(data.recap?[["recap",recapScene(data)]]:[])]) {
    const label=`${filename}/${id}`;
    inspectScene(layout,label);pilotScenes.push({layout,label});
  }
}
for(let n=4;n<=6;n++) for(let target=0;target<n;target++) {
  const g=geometry({steps:Array.from({length:n},()=>({})),feedback:"repeat",feedback_to:target});
  assert(boundary(g.feedback.points[0],g.nodes.at(-1))); assert(boundary(g.feedback.points.at(-1),g.nodes[target]));
}
// Generic exported schematics include circles and labels aligned left.
const fixture={width:1440,height:530,items:[
  {kind:"block",shape:"ellipse",x:20,y:40,w:50,h:50,fill:"FFFFFF",color:"009E73"},
  {kind:"text",text:"addition",x:100,y:40,w:240,h:35,size:24,color:"1F2937",bold:false,align:"left"},
  {kind:"route",points:[[70,65],[95,65]],color:"009E73"},
]};
inspectScene(fixture,"generic scene");
assert(svgScene(fixture).includes("<ellipse "));assert(svgScene(fixture).includes('text-anchor="start"'));
const ellipseShape=nativeItems((s,a)=>nativeScene(s,fixture,a,{x:0,y:0,w:1440,h:530})).shapes.find(s=>s.kind==="ellipse");
assert.equal(ellipseShape.rectRadius,undefined,"Ellipse must not carry a rounded-rectangle adjustment: Office rejects that deck");
const selfLoop=geometry({layout:"linear",steps:Array.from({length:5},()=>({})),arrows:["a","b","c","d"],feedback:"repeat",feedback_from:3,feedback_to:3});
assert(boundary(selfLoop.feedback.points[0],selfLoop.nodes[3]));
assert(boundary(selfLoop.feedback.points.at(-1),selfLoop.nodes[3]));
assert.notEqual(selfLoop.feedback.points[1][0],selfLoop.feedback.points[2][0],"Self-repeat needs a visible horizontal rail");
(async()=>{
  for(const {layout,label} of pilotScenes) for(const i of layout.items.filter(i=>i.kind==="text")) {
    const escaped=i.text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const measured=await sharp({text:{text:escaped,font:`Arial ${i.bold?"Bold ":""}${i.size}`,dpi:72,rgba:true}}).metadata();
    assert(measured.width<=i.w+2,`${label}: actual Arial width exceeds box: ${i.text}`);
  }
  console.log(`Technical diagrams: ${count} authored routes, ${linear} pilot scenes with measured Arial text widths, 15 legacy feedback cases and generic scene parity passed`);
})().catch(e=>{console.error(e);process.exitCode=1;});
