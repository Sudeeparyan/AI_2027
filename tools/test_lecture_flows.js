"use strict";
// Strict operation schema, cross-format scene parity and layout bounds.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {flowDiagram,scene,flattenedScene,svgScene,ROLES} = require("./technical_diagrams");
const {layouts} = require("./slides");
const clone = value => JSON.parse(JSON.stringify(value));
const flow = {type:"flow",title:"Trace the operations",steps:[
  {label:"Read",role:"data"},{label:"Transform",sub:"Use weights",role:"model"},
  {label:"Report",role:"output"}],arrows:["inputs","scores"]};
function recorder() {
  const events=[];
  return {events,slide:{addText:(text,options)=>events.push({kind:"text",text,options}),
    addShape:(shape,options)=>events.push({kind:"shape",shape,options}),
    addImage:options=>events.push({kind:"image",options})}};
}
function bounded(events) {
  for (const e of events) {
    const o=e.options;
    assert(o.x>=-.001 && o.y>=1.62-.001 && o.x+(o.w||0)<=13.3334 && o.y+(o.h||0)<=6.901,
      `flow item outside panel: ${e.text || e.shape}`);
    if(o.fontSize !== undefined) assert(o.fontSize>=14-.001,"Flow font below 14 pt");
    for(const r of Array.isArray(e.text)?e.text:[]) if(r.options?.fontSize!==undefined)
      assert(r.options.fontSize>=14,"Flow run below 14 pt");
  }
}
function sceneChecks(g) {
  const xml=svgScene(g);
  for(const t of flattenedScene(g).items.filter(i=>i.kind==="text"))
    assert(xml.includes(t.text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")),"SVG lost scene label");
  for(const i of g.items) if(i.kind!=="route") {
    assert(i.x>=0 && i.y>=0 && i.x+i.w<=g.width+.01 && i.y+i.h<=g.height+.01,"Scene item outside canvas");
    if(i.kind==="text") assert(i.text.split("\n").filter(Boolean).length*i.size*1.16<=i.h+.01);
  }
}
(async()=>{
  const variants=[flow,{...flow,bullets:["Predict the output."],callout:"Weights stay fixed."},
    {...flow,feedback:"repeat",feedback_from:1,feedback_to:1},
    {type:"flow",title:"Combine independent inputs",steps:[
      {label:"Read",role:"data"},{label:"Draw",role:"data"},{label:"Combine",role:"model"}],
      arrows:[null,"noise"],branches:[{from:0,to:2,label:"input"}]}];
  for(const variant of variants) {
    const g=scene(flowDiagram(variant));sceneChecks(g);
    const a=recorder(),b=recorder();
    await layouts.flow(a.slide,variant,{assets:{figures:{}}});
    await layouts.flow(b.slide,{...variant,flow_asset:"lecture_flow_01"},{assets:{figures:{lecture_flow_01:{scene:g}}}});
    assert.deepEqual(a.events,b.events,"Exported and direct scene rendering differ");
    assert(!a.events.some(e=>e.kind==="image"),"Flow was rasterised in PowerPoint");
    bounded(a.events.slice(1)); // exclude the slide title above the body panel
    const labels=a.events.filter(e=>e.kind==="text"&&typeof e.text==="string").map(e=>e.text);
    for(const t of flattenedScene(g).items.filter(i=>i.kind==="text")) assert(labels.includes(t.text),"Native scene lost a label");
  }
  for(const role of Object.keys(ROLES)) {
    const value=clone(flow);value.steps[1].role=role;
    const g=scene(flowDiagram(value));
    assert(g.items.some(i=>i.kind==="text"&&i.text===ROLES[role].label));
  }
  for(const n of [2,7]) {
    const value={type:"flow",title:"Valid length",steps:Array.from({length:n},()=>({label:"Run",role:"model"})),arrows:Array(n-1).fill("data")};
    sceneChecks(scene(flowDiagram(value)));
  }
  for(const mutate of [v=>v.steps.pop(),v=>v.arrows[0]="",v=>delete v.steps[1].role,
    v=>v.steps[1].role="constructor",v=>v.steps[1].label="",v=>v.steps[1].sub=42,
    v=>v.arrows=[null,null],v=>v.branches=[{from:0,to:7,label:"input"}],v=>v.feedback="",
    v=>{v.feedback="repeat";v.feedback_to=9;}]) {
    const value=clone(flow);mutate(value);assert.throws(()=>flowDiagram(value));
  }
  const excessive=clone(flow);excessive.steps[0].sub="word ".repeat(41);
  await assert.rejects(layouts.flow(recorder().slide,excessive),/40|too long|does not fit/);
  const owned=clone(flow);owned.steps[1].code="transform";owned.steps[1].code_owner=true;
  assert(svgScene(scene(flowDiagram(owned))).includes("transform"));
  delete owned.steps[1].code_owner;
  assert(!svgScene(scene(flowDiagram(owned))).includes("transform"));
  // A CLI fixture passed by the Python integration test exercises exported
  // pixels and the actual authored pilot flows without a weekly build.
  if(process.argv[2]) {
    for(const {id,slide} of JSON.parse(fs.readFileSync(process.argv[2],"utf8")).flows) {
      const r=recorder();await layouts.flow(r.slide,slide);bounded(r.events.slice(1));
      console.log(`${id}: authored flow panel/font/body checks passed`);
    }
  }
  for(const count of [3,4,6]) {
    const r=recorder();await layouts.cards(r.slide,{type:"cards",title:"Examples",cards:Array.from({length:count},(_,i)=>({head:`Example ${i+1}`,text:"Input becomes output."}))});
    assert(!r.events.some(e=>e.kind==="image" || e.shape==="ellipse"),"Unrequested lightbulb badge remains");
    assert(r.events.filter(e=>e.kind==="text").length>=count*2,"Card examples lost their text");
  }
  console.log("Lecture flows: strict roles/arrows/branches, portable/native parity, word/font/panel limits and icon-free cards passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
