// Course map and class recap share exact geometry in editable slides, SVG and PNG.
"use strict";
const fs = require("node:fs"), path = require("node:path"), sharp = require("sharp");
const {svgScene, fitLines, ROLES} = require("./technical_diagrams");
const INK="1E1B4B", TEXT="1F2937", W=1440;
function text(items, content, x,y,w,h,size=30,bold=false,color=TEXT,align="left") {
  const rows=fitLines(content,w,size,bold);
  if(rows.length*size*1.16>h+1) throw Error(`Support text too tall: ${content}`);
  items.push({kind:"text",text:rows.join("\n"),x,y,w,h,size,bold,color,align});
}
function wrap(content,limit) {
  const result=[]; let row="";
  for(const word of String(content).split(/\s+/)) {
    if(row && (row+" "+word).length>limit) {result.push(row);row=word;}
    else row+=(row?" ":"")+word;
  }
  if(row) result.push(row);
  return result.join("\n");
}
function recapScene(data) {
  const recap=data.recap, items=[], cards=recap.cards;
  if(cards.length<4 || cards.length>6) throw Error("A recap needs four to six concepts");
  const rows=2, cols=3, gap=22, margin=28, cw=(W-2*margin-2*gap)/cols, ch=185;
  cards.forEach((card,i)=>{
    const x=margin+(i%cols)*(cw+gap), y=20+Math.floor(i/cols)*(ch+gap), role=ROLES[card.role||"model"];
    if(!role) throw Error("Unknown recap role "+card.role);
    items.push({kind:"block",shape:role.shape,x,y,w:cw,h:ch,fill:role.fill,color:role.color});
    text(items,card.title,x+18,y+6,cw-36,72,30,true,INK);
    text(items,card.text,x+18,y+81,cw-36,94,26,false,TEXT);
  });
  text(items,"Example to explain aloud",margin,430,W-2*margin,35,26,true,INK);
  text(items,recap.example,margin,472,W-2*margin,63,27);
  // "In Week 3 we ..." reads as "Next: in Week 3 we ..."; keep capitals such as "LLMs".
  const next=recap.next.replace(/^Next:\s*/i,"").replace(/^[A-Z](?=[a-z])/,c=>c.toLowerCase());
  text(items,"Next: "+next,margin,544,W-2*margin,62,26,false,TEXT);
  return {width:W,height:612,items};
}
const WEEK_LABELS=[
  "Introduction and\nresponsible AI","Variational\nautoencoders (VAEs)","Generative adversarial\nnetworks (GANs)","Diffusion\nmodels",
  "Transformers\nand attention","Language models\nand families","Prompting\nand evaluation","Fine-tuning\nand adaptation",
  "Multimodal\nfoundations","Multimodal\napplications","Deployment\nand frameworks","Retrieval (RAG)\nand agents",
];
function courseMapScene(week) {
  const items=[], margin=36,gap=38,cw=(W-2*margin-gap*3)/4,ch=104,inset=14;
  const themes=["Foundations and image generators","Language models and adaptation","Multimodal applications and deployed systems"];
  for(let row=0;row<3;row++) {
    const y=32+row*187;
    text(items,themes[row],margin,y,W-2*margin,36,27,true,INK);
    for(let col=0;col<4;col++) {
      const i=row*4+col,x=margin+col*(cw+gap),top=y+46,current=i+1===week;
      const fill=current?"FFF0E6":"E4F2FB",color=current?"D55E00":"0072B2";
      items.push({kind:"block",shape:"roundRect",x,y:top,w:cw,h:ch,fill,color});
      text(items,"Week "+(i+1)+(current?": you are here":""),x+inset,top+2,cw-2*inset,32,25,true,color);
      text(items,WEEK_LABELS[i],x+inset,top+38,cw-2*inset,64,25,true,INK);
      if(col<3) items.push({kind:"route",points:[[x+cw,top+ch/2],[x+cw+gap,top+ch/2]],color:"0072B2"});
    }
    if(row<2) {
      // Continue from the last week of this row to the first week of the next row.
      const lastX=margin+3*(cw+gap)+cw/2, bottom=y+46+ch, rail=bottom+18, nextMid=y+187+46+ch/2;
      items.push({kind:"route",points:[[lastX,bottom],[lastX,rail],[margin/2,rail],[margin/2,nextMid],[margin,nextMid]],color:"0072B2"});
    }
  }
  text(items,"Responsible use and evaluation continue throughout the course.",margin,601,W-2*margin,37,27,false,TEXT);
  return {width:W,height:658,items};
}
async function main() {
  const [source,out]=process.argv.slice(2),data=JSON.parse(fs.readFileSync(source,"utf8"));
  fs.mkdirSync(out,{recursive:true});
  for(const [id,g] of [["course_map",courseMapScene(data.week)],...(data.recap?[["recap",recapScene(data)]]:[])]) {
    const stem=path.join(out,"beginner_"+id),xml=svgScene(g,id==="recap"?data.recap.title:"The twelve-week course map");
    fs.writeFileSync(stem+".scene.json",JSON.stringify(g,null,2));
    fs.writeFileSync(stem+".svg",xml);
    await sharp(Buffer.from(xml)).resize({width:2160}).png().toFile(stem+".png");
  }
}
module.exports={recapScene,courseMapScene};
if(require.main===module) main().catch(e=>{console.error(e);process.exitCode=1;});
