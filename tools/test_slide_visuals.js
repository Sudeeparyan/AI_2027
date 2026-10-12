// Informative editable examples and prediction-before-answer regressions.
"use strict";
const assert = require("node:assert/strict");
const {layouts, validateVisualSlide} = require("./slides");
function recorder() {
  const events=[];
  return {events, slide:{addText:(text,options)=>events.push({kind:"text",text,options}),
    addShape:(shape,options)=>events.push({kind:"shape",shape,options}),
    addImage:options=>events.push({kind:"image",options}),
    addTable:(rows,options)=>events.push({kind:"table",rows,options})}};
}
const example={type:"bullets",title:"Tokens and vectors",bullets:["IDs select learned vectors.","Shape records rows and width."],
  icon:"fa:FaTable",visual:{type:"example_strip",items:[
    {label:"Text",text:"cat"},{label:"IDs",text:"[2, 0, 1]"},{label:"Vectors",text:"[3, 8]"}]}};
const table={type:"bullets",title:"Compare shapes",bullets:["One vector per token."],visual:{type:"mini_table",
  header:["Input","Output"],rows:[["3 IDs","[3, 8]"],["4 IDs","[4, 8]"]]}};
const textContent=events=>events.filter(e=>e.kind==="text").flatMap(e=>Array.isArray(e.text)?e.text.map(r=>r.text):[e.text]).join(" ");
(async()=>{
  const ex=recorder(); await layouts.bullets(ex.slide,example,{assets:{figures:{}}});
  assert(!ex.events.some(e=>e.kind==="image"),"An authored example must replace the decorative icon");
  assert(textContent(ex.events).includes("[3, 8]"),"The worked tensor shape must remain editable text");
  const tb=recorder(); await layouts.bullets(tb.slide,table,{assets:{figures:{}}});
  assert.equal(tb.events.filter(e=>e.kind==="table").length,1,"Comparison must use an editable table");
  assert.equal(tb.events.find(e=>e.kind==="table").rows.length,3,"Header and all evidence rows must survive");
  for(const e of [...ex.events,...tb.events]) {
    const o=e.options;
    assert(o.x>=0 && o.y>=0 && o.x+(o.w||0)<=13.333+0.001 && o.y+(o.h||0)<=7.5,"Example outside slide bounds");
    if(o.fontSize!==undefined) assert(o.fontSize>=14,"Text below projected minimum");
    for(const run of Array.isArray(e.text)?e.text:[]) if(run.options?.fontSize!==undefined)
      assert(run.options.fontSize>=14,"Run below projected minimum");
  }
  assert.throws(()=>validateVisualSlide({...example,bullets:[{text:"A",sub:["B","C","D"]}]}),/three bullets/);
  assert.throws(()=>validateVisualSlide({...example,bullets:["word ".repeat(41)]}),/40/);
  assert.throws(()=>validateVisualSlide({...table,visual:{...table.visual,rows:[["missing cell"]]}}),/matching string rows/);
  assert.throws(()=>validateVisualSlide({...example,visual:{type:"decorative_icon"}}),/Unknown visual/);
  const question={type:"quiz",title:"Predict",question:"Which shape has three rows?",options:["[3, 8]","[8, 3]"],
    notes:"Answer: A. Rows count tokens."};
  const a=recorder(),b=recorder();
  await layouts.quiz(a.slide,{...question,answer:"A",reveal:true,show_answer:true});
  await layouts.quiz(b.slide,{...question,answer:"B",reveal:false,show_answer:false});
  assert.deepEqual(a.events,b.events,"Changing correctness/reveal metadata must never change the visible question");
  assert(!textContent(a.events).includes("Answer:"),"Instructor answer leaked onto prediction slide");
  assert(textContent(a.events).includes("[3, 8]") && textContent(a.events).includes("[8, 3]"));
  console.log("Slide visuals: editable examples/table, body limits, bounds, fonts and answer-hidden prediction passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
