"""Compile the authored pedagogy into 40 editable slide records per week.

Run intentionally after editing author_lessons.py or weeks.json. Ordinary course.py
builds do not overwrite slides.json, so manual edits to a slide survive iteration.
"""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
weeks=read(D/'weeks.json')

def notes(kind,w,body,extra):
    case=w['case'];goal=w['goal']
    prompts={
      'open':'Show the visual for ten seconds, then ask for an initial decision. Do not reveal the answer yet. Ask a student to name the source or measurement that would change their mind.',
      'question':'Take a show of hands for each choice before showing the mechanism. Ask pairs to defend a choice with evidence. Save the strongest wrong reason for the reveal later in this cycle.',
      'analogy':'Begin with the familiar scene. Invite learners to describe each part in everyday language. Then ask where the comparison might fail before introducing the technical vocabulary.',
      'mapping':'Point from each familiar element to the corresponding technical element. Have students cover the right column and reconstruct the three matches. Read the limitation aloud.',
      'mechanism':'Trace every arrow slowly from the input to the output. Ask what information crosses the boundary at each arrow. Pause if students confuse a simplified notebook with a trained model.',
      'worked':'Work the specific numbers or source wording aloud. Give students thirty seconds to write a conclusion, then ask which source line or calculation supports it.',
      'reveal':'Return to the earlier choice. Explain the correct answer and why the tempting claim fails. Ask one learner to say the answer in a single sentence without reading it.',
      'practice':'Open the matching notebook and demonstrate the first run. Have learners predict one change, run only that change, and record input, setting, output and an error explanation.',
      'check':'Use this as a brief formative check. Ask for the conclusion, the evidence and one limitation. Revisit disagreements in the review session and offer an extension only after the core task.',
    }
    return (f'TEACHER SCRIPT — {kind.upper()}. {prompts[kind]} This week’s case is: {case} '
            f'The visible slide states: {body} {extra} The target is: {goal} '
            'Listen for an explanation of the actual arrow or calculation, not just a repeated label. '
            'If the answer is unsupported, mark that uncertainty and return to the source. '
            'Before moving on, ask for the main idea and one concrete change to the worked example.')

def add(slides,w,title,body,kind,diagram,extra):
    i=len(slides)+1
    slides[f's{i:02d}']={'title':title,'explanation':body,'speaker_notes':notes(kind,w,body,extra),'diagram':diagram,'teaching_stage':kind}

def compile_week(n):
    path=R/'content'/f'week_{n:02d}';w=read(path/'lesson.json');v=read(path/'visuals.json');st=read(path/'study.json');d=weeks[str(n)];slides={}
    add(slides,w,f'Week {n:02d} · {w["title"]}',d['story']+'\n\n'+d['hook'],'open',{'type':'story','items':[d['story'],'What is known?','What must we check?']},'Reveal that the case will be revisited after four short concept cycles.')
    add(slides,w,'First vote: what would you do?',d['hook']+'\n\nChoose an initial answer and name one piece of evidence you need.','question',{'type':'quiz','items':['Accept the first plausible result','Check the source or measurement']},'Accept reasonable uncertainty; record the initial votes to revisit at the end.')
    add(slides,w,'The journey through today’s case','We will predict, connect an analogy, trace a mechanism, work an example, then explain a decision.','open',{'type':'steps','items':[t[0] for t in w['terms']]},'Point to the four named ideas, then show where the notebook experiment fits.')
    for j,((term,definition),c) in enumerate(zip(w['terms'],d['concepts']),1):
        add(slides,w,f'{j}. Predict · {term}',c['question']+'\n\nMake a choice before we explain the diagram.','question',{'type':'quiz','items':[c['answer'],c['wrong']]},f'The intended answer is {c["answer"]} The tempting claim is {c["wrong"]} Do not show which card is correct until the reveal.')
        add(slides,w,f'{term} · a familiar picture',c['analogy']+'\n\nWhat matches our case, and where does the analogy stop?','analogy',{'type':'analogy','items':[x[0] for x in c['mapping']],'caption':c['analogy']},f'The boundary is: {c["limit"]}')
        add(slides,w,f'{term} · map the parts','Match each everyday part to the technical part.\n\nAnalogy limit: '+c['limit'],'mapping',{'type':'map','pairs':c['mapping']},f'The three precise mappings are {c["mapping"]}.')
        add(slides,w,f'{term} · how it works',definition+'\n\nTrace the arrows from left to right.','mechanism',{'type':'flow','items':v['term'][j-1]},f'The weekly example is: {c["worked"]} State what the analogy does not explain: {c["limit"]}')
        add(slides,w,f'{term} · work one case',c['worked']+'\n\nPoint to the evidence before giving your conclusion.','worked',{'type':'work','items':[c['worked'],c['question'],c['answer']]},f'The expected reasoning concludes: {c["answer"]} Distinguish measured results from illustrative numbers.')
        add(slides,w,f'{term} · answer and explain',c['answer']+'\n\nTempting mistake: '+c['wrong'],'reveal',{'type':'reveal','items':[c['wrong'],c['answer']],'caption':c['limit']},f'The earlier question was {c["question"]} Ask how this differs from the wrong claim.')
    assert len(slides)==27
    add(slides,w,'Evidence trail: input → result',w['example'][1]+'\n\n'+w['example'][2],'worked',{'type':'work','items':w['example']},'Have one student point to an exact source word or number for each conclusion.')
    add(slides,w,'Notebook: run the prepared example',w['lab']+'\n\nRead the original output before editing.','practice',{'type':'steps','items':w['steps']},f'Use Week_{n:02d}_Lab.ipynb. This lab is a prepared classroom demonstration with its limitations documented.')
    add(slides,w,'Find the input and check in Python','Locate the supplied input, the transformation, the checking line and the printed result.','practice',{'type':'flow','items':['Input data','Python operation','Check','Recorded result']},f'The weekly lab change is: {w["change"]}')
    add(slides,w,'Predict one controlled change',w['change']+'\n\nWhat should change in the result? What stays fixed?','practice',{'type':'compare','left':'Before: original case','right':'After: one change','caption':'Keep all other inputs fixed.'},'Ask learners to write their prediction before rerunning the cell.')
    add(slides,w,'Run, compare and record','Record the input, the one changed setting, the observed result and a reason for any difference.','practice',{'type':'table','rows':[['Input','Setting','Result'],['Original','baseline','record'],['Changed','one edit','compare']]},f'The required evidence is: {w["output"]}')
    add(slides,w,'Measure without hiding the denominator',w['metric']+'\n\nWrite the numerator, denominator and one failure case.','check',{'type':'flow','items':['Define success','Count successes','Count cases','Inspect errors']},f'Where a measure is qualitative, show the concrete source comparison. Example: {w["example"][2]}')
    add(slides,w,'Error detective: find the layer','Plausible wrong claim: '+w['misconception']+'\n\nCorrect it using the case evidence.','check',{'type':'reveal','items':[w['misconception'],w['correction']],'caption':'Name a test that would catch this failure.'},f'The correction is {w["correction"]}')
    add(slides,w,'Transfer to a new situation',d['transfer']+'\n\nExplain which source or test would settle it.','check',{'type':'quiz','items':['Use the visible evidence','Flag missing or unsafe evidence']},'This is a fresh scenario, so ask for a reason before accepting either short answer.')
    add(slides,w,'Explain it to a partner','Without reading the slide, name the input, one processing step and the check. Your partner should ask “How do you know?”','check',{'type':'flow','items':[v['path'][0],v['path'][1],v['path'][-1]]},'Listen for a causal explanation and one limitation, then revisit a difficult arrow on the board.')
    for q in st['exam'][:2]:
        add(slides,w,'Exam rehearsal · explain and apply',q['question']+'\n\nWrite a short answer, then compare with the guide.','check',{'type':'steps','items':['State the core idea','Use a case detail or calculation','Add a limitation','Compare with the Word guide']},f'Model answer after discussion: {q["model_answer"]} Practice marks: {q["marks"]}.')
    add(slides,w,'What to hand in',w['output']+'\n\nInclude the one changed input and an error explanation.','practice',{'type':'steps','items':['Original run','One changed input','Comparison','Explanation + source']},f'This weekly evidence receives formative feedback according to the module sequence; refer to the assessment brief for graded work.')
    add(slides,w,'Three ideas to recall next week',d['memory']+'\n\nRead the linked official sources in the Word guide and redraw one diagram from memory.','check',{'type':'steps','items':[w['terms'][0][0],w['terms'][1][0],w['terms'][2][0],w['terms'][3][0]]},f'Return to the opening vote and ask whether the evidence changed anyone’s mind. The next week builds on this outcome: {w["goal"]}')
    assert len(slides)==40,(n,len(slides))
    (path/'slides.json').write_text(json.dumps(slides,indent=2,ensure_ascii=False),encoding='utf-8')
    return {'week':n,'slides':len(slides),'concepts':len(d['concepts'])}

if __name__=='__main__':
    print(json.dumps([compile_week(n) for n in range(1,13)]))
